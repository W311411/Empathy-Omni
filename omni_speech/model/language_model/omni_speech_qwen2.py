from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from ..omni_speech_arch import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM
import pdb
import time

class OmniSpeechConfig(Qwen2Config):
    model_type = "omni_speech_qwen"

class OmniSpeechQwen2Model(OmniSpeechMetaModel, Qwen2Model):
    config_class = OmniSpeechConfig

    def __init__(self, config: Qwen2Config):
        super(OmniSpeechQwen2Model, self).__init__(config)
        
class OmniSpeechQwen2ForCausalLM(Qwen2ForCausalLM, OmniSpeechMetaForCausalLM):
    config_class = OmniSpeechConfig

    def __init__(self, config: Qwen2Config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = OmniSpeechQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        wav: Optional[torch.FloatTensor] = None,
        wav_lengths: Optional[torch.LongTensor] = None,
        emotion_wavs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                out_emotion_features
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths,
                wav,
                wav_lengths,
                emotion_wavs
            )

        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )
        # 关键优化：向量化处理emotion loss，避免逐batch循环
        if hasattr(result, 'hidden_states') and result.hidden_states is not None:
            hidden = result.hidden_states[-1]
            result.hidden_states = None  # 立即释放其他层
            emotion_loss = self._compute_emotion_loss(hidden, labels, out_emotion_features)
            
            if emotion_loss.numel() > 0:
                llm_loss = result.loss.item() if result.loss is not None else 0.0
                result.loss = result.loss + emotion_loss
                
                # 降低打印频率，减少同步开销
                if torch.rand(1).item() < 0.1:  # 从0.1降到0.01
                    print(f"Emotion loss: {emotion_loss.item():.4f}, LLM loss: {llm_loss:.4f}")

        return result
    
    def _compute_emotion_loss(self, hidden, labels, out_emotion_features):
        """
        计算emotion loss
        Args:
            hidden: 模型的hidden states [batch_size, seq_len, hidden_dim]
            labels: 标签张量 [batch_size, seq_len]
            out_emotion_features: emotion features [batch_size, emotion_dim]
        Returns:
            emotion loss
        """
        batch_size = labels.size(0)
        ignore_mask = (labels != -100)
        
        batch_hidden_avg = []
        valid_batch_indices = []
        
        for batch_idx in range(batch_size):
            valid_pos = ignore_mask[batch_idx].nonzero(as_tuple=True)[0]
            
            if len(valid_pos) > 0:
                top5_pos = valid_pos[:5]
                hidden_avg = hidden[batch_idx, top5_pos].mean(dim=0)  # [hidden_dim]
                batch_hidden_avg.append(hidden_avg)
                valid_batch_indices.append(batch_idx)
        
        if not batch_hidden_avg:
            return torch.tensor(0.0, device=hidden.device)
        
        batch_hidden_avg = torch.stack(batch_hidden_avg, dim=0)
        
        valid_emotion_features = out_emotion_features[valid_batch_indices]  # [valid_batch_size, emotion_dim]
        
        predictions = self.get_emotion_predict_head()(batch_hidden_avg)  # [valid_batch_size, emotion_dim]
        
        l2_loss = nn.functional.mse_loss(predictions, valid_emotion_features)
        cosine_sim = nn.functional.cosine_similarity(predictions, valid_emotion_features, dim=-1)
        cosine_loss = 1 - cosine_sim.mean()
        
        total_loss = 0.6 * l2_loss + 0.4 * cosine_loss
        return total_loss

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        wav: Optional[torch.Tensor] = None,
        wav_lengths: Optional[torch.Tensor] = None,
        emotion_wavs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                out_emotion_features
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths,
                wav,
                wav_lengths,
                emotion_wavs
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        result = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            **kwargs
        )
        # pdb.set_trace()
        hidden = result.hidden_states[-1]
        del result.hidden_states
        emotion = self.get_model().emotion_head(out_emotion_features)
        return result.sequences, emotion

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs
    def _post_decode(self, output, temperature=1.0, top_k=0, top_p=0.0):
        """
        Decoding function, based on the posterior probability output, 
        uses top_k, top_p, and temperature parameters for sampling.

        Parameters:
        - output: torch.Tensor, shaped as (1, 1, D), represents the posterior probability output by the model.
        - top_k: int, indicates selecting the top k tokens with the highest probability for sampling.
                      If 0, no top_k filtering is performed.
        - top_p: float, indicates selecting tokens with cumulative probability not exceeding p for sampling.
                        If 0.0, no top_p filtering is performed.
        - temperature: float, represents the sampling temperature parameter. 
                              The higher the value, the more random the sampling; 
                            the lower the value, the more deterministic the sampling.

        Returns:
        - Selected token index.
        """
        output = output.squeeze(0).squeeze(0)

        # temperature
        if temperature != 1.0:
            output = output / temperature

        probs = torch.nn.functional.softmax(output, dim=-1)

        # top_k
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
            probs = probs / probs.sum()

        # top_p
        if top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove[0]:
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()

        token_index = torch.multinomial(probs, 1)
        return token_index.unsqueeze(0)
    
    def _generate_one_step(
        self,
        inputs_embeds: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.1,
        **kwargs
    ) -> Tuple[torch.LongTensor, List[torch.FloatTensor]]:
        """
        Generates the model's next output based on the current input and state.

        Parameters:
        - inputs: The input tensor containing the model's input data.
        - stat: The current state information used to control the generation process.
        - top_p: The threshold for controlling top-p sampling.
        - top_k: The threshold for controlling top-k sampling.
        - temperature: Controls the randomness of sampling.

        Returns:
        - last_id: The index of the last generated token.
        - stat: The updated state information.
        - past_key_values: The model's historical key-value pairs, used for cross-step memory.
        - hidden_state: The model's hidden state, used to maintain cross-step contextual information.
        """
        outputs = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs
        )
        last_hidden_state = outputs['hidden_states'][-1]
        last_logit = self.lm_head(last_hidden_state[:, -1:, :])
        last_id = self._post_decode(last_logit, temperature=temperature, top_k=top_k, top_p=top_p)
        return_tts_state = last_hidden_state[:, -1:, :]
        return last_id, outputs['past_key_values'], return_tts_state


AutoConfig.register("omni_speech_qwen", OmniSpeechConfig)
AutoModelForCausalLM.register(OmniSpeechConfig, OmniSpeechQwen2ForCausalLM)
