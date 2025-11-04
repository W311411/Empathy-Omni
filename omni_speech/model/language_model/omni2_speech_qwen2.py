#    Copyright 2023 Haotian Liu
#    Copyright 2024 Qingkai Fang
#
#    This project is modified based on LLaVA by Haotian Liu, Qingkai Fang adds further supports for speech-to-text/speech tasks.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..omni_speech_arch2 import Omni2SpeechMetaModel, Omni2SpeechMetaForCausalLM


class Omni2SpeechQwen2Config(Qwen2Config):
    model_type = "omni2_speech_qwen2"


class Omni2SpeechQwen2Model(Omni2SpeechMetaModel, Qwen2Model):
    config_class = Omni2SpeechQwen2Config

    def __init__(self, config: Qwen2Config):
        super(Omni2SpeechQwen2Model, self).__init__(config)


class Omni2SpeechQwen2ForCausalLM(Qwen2ForCausalLM, Omni2SpeechMetaForCausalLM):
    config_class = Omni2SpeechQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Omni2SpeechQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
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
        emotion_lengths: Optional[torch.LongTensor] = None,
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
                emotion_wavs,
                emotion_lengths,
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

        if hasattr(result, 'hidden_states') and result.hidden_states is not None:
            hidden = result.hidden_states[-1]
            result.hidden_states = None 
            emotion_loss = self._compute_emotion_loss(hidden, labels, out_emotion_features)
            hidden = None 

            if emotion_loss.numel() > 0:
                llm_loss = result.loss.item() if result.loss is not None else 0.0
                result.loss = result.loss + emotion_loss
                
                if torch.rand(1).item() < 0.1: 
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
        # 计算loss
        l2_loss = nn.functional.mse_loss(predictions, valid_emotion_features)
        cosine_sim = nn.functional.cosine_similarity(predictions, valid_emotion_features, dim=-1)
        cosine_loss = 1 - cosine_sim.mean()
        # 组合loss
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

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

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

AutoConfig.register("omni2_speech_qwen2", Omni2SpeechQwen2Config)
AutoModelForCausalLM.register(Omni2SpeechQwen2Config, Omni2SpeechQwen2ForCausalLM)
