import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Config
from omni_speech.constants import IGNORE_INDEX


def lengths_to_attention_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return ~mask

class AdaLN(nn.Module):
    def __init__(self, normalized_shape, condition_dim, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.condition_projection = nn.Linear(condition_dim, normalized_shape * 2)
        
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)
        nn.init.ones_(self.condition_projection.bias[:normalized_shape])
    
    def forward(self, x, condition):
        """
        Args:
            x: 输入序列，支持2D [seq_len, hidden_size] 或 3D [batch_size, seq_len, hidden_size]
            condition: 条件特征，对应维度
        """
        original_shape = x.shape
        is_2d_input = len(original_shape) == 2
        
        # 如果是2D输入，添加batch维度
        if is_2d_input:
            x = x.unsqueeze(0)  # [seq_len, hidden] -> [1, seq_len, hidden]
            condition = condition.unsqueeze(0)  # [seq_len, condition_dim] -> [1, seq_len, condition_dim]
        
        # 标准 Layer Normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 从条件特征生成 scale 和 shift
        condition_params = self.condition_projection(condition)
        scale, shift = condition_params.chunk(2, dim=-1)
        
        # 不再需要判断维度，直接应用
        output = normalized * scale + shift
        
        # 如果输入是2D，移除batch维度
        if is_2d_input:
            output = output.squeeze(0)  # [1, seq_len, hidden] -> [seq_len, hidden]
        
        return output

class LLMSpeechGenerator(nn.Module):
    def __init__(self, config):
        super(LLMSpeechGenerator, self).__init__()
        self.model = Qwen2ForCausalLM(Qwen2Config(**config.speech_generator))
        self.tokenizer = AutoTokenizer.from_pretrained(config.tts_tokenizer)
        self.input_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, self.model.config.hidden_size)
        )
        self.stream_params = config.stream_params
        self.gate = nn.Sequential(
            nn.Linear(2 * self.model.config.hidden_size, self.model.config.hidden_size),
            nn.Sigmoid()
        )
        self.adaln = AdaLN(config.hidden_size, 768)

    def fusion(self, rep, emb):
        gate = self.gate(torch.cat([rep, emb], dim=-1))
        return rep * gate + emb * (1 - gate)
    
    def forward(self, llm_hidden_list, llm_labels, speech_tokens, txt_eos_emb=None, emotion_list=None):
        batch_size = len(llm_hidden_list)
        llm_hidden_filter_list = []
        llm_hidden_lens = []
        for llm_rep, llm_label, emotion_feat in zip(llm_hidden_list, llm_labels, emotion_list):
            # llm_hidden_filter_list.append(llm_rep[llm_label != IGNORE_INDEX])
            llm_hidden_filter = llm_rep[torch.logical_and(llm_label != IGNORE_INDEX, llm_label != 128009)]
            if txt_eos_emb is not None:
                llm_hidden_filter = self.adaln(llm_hidden_filter[:-1], emotion_feat)
                llm_hidden_filter = torch.cat([llm_hidden_filter, txt_eos_emb.squeeze(0)], dim=0)
            llm_hidden_filter_list.append(llm_hidden_filter)
            llm_hidden_lens.append(llm_hidden_filter_list[-1].shape[0])
        llm_hidden_lens = torch.tensor(llm_hidden_lens).to(llm_hidden_filter_list[0].device)

        # llm_hidden_list = [llm_hidden[torch.logical_and(llm_label != IGNORE_INDEX, llm_label != 128009)] for llm_hidden, llm_label in zip(llm_hidden_list, llm_labels)]

        max_len = max([rep.size(0) for rep in llm_hidden_filter_list])
        llm_hidden_states = torch.zeros(len(llm_hidden_filter_list), max_len, llm_hidden_filter_list[0].size(1), device=llm_hidden_filter_list[0].device, dtype=llm_hidden_filter_list[0].dtype)
        for i, rep in enumerate(llm_hidden_filter_list):
            llm_hidden_states[i, :rep.size(0), :] = rep

        bos_token = torch.full((batch_size, 1), self.bos_token, dtype=torch.long, device=llm_hidden_states.device)
        sos_token = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=llm_hidden_states.device)
        eos_token = torch.full((batch_size, 1), self.eos_token, dtype=torch.long, device=llm_hidden_states.device)
        padding_token = torch.full((batch_size, 1), self.padding_token, dtype=torch.long, device=llm_hidden_states.device)

        speech_tokens[speech_tokens == IGNORE_INDEX] = self.padding_token
        speech_tokens_lens = []
        for tgt_unit in speech_tokens:
            speech_tokens_lens.append(torch.sum(tgt_unit != self.padding_token))
        speech_tokens_lens = torch.tensor(speech_tokens_lens).to(llm_hidden_filter_list[0].device)

        # # # start forwarding
        # pdb.set_trace()
        llm_hidden_states = self.pre_nn_forward(llm_hidden_states, llm_hidden_lens)
        bos_emb = self.embedding(bos_token)
        llm_hidden_states = torch.cat([bos_emb, llm_hidden_states], dim=1)
        llm_hidden_lens = llm_hidden_lens + 1

        # Create input x with sos token at the beginning
        speech_max_len = speech_tokens.shape[1]
        in_speech_tokens = torch.cat([sos_token, speech_tokens], dim=1) 
        
        # Create output y with eos token at the end
        out_speech_tokens = torch.cat([speech_tokens, padding_token], dim=1)
        eos_positions = torch.arange(speech_max_len + 1, device=speech_tokens.device).expand(batch_size, speech_max_len + 1) == speech_tokens_lens.unsqueeze(1)
        out_speech_tokens = out_speech_tokens.masked_scatter(eos_positions, eos_token.expand_as(out_speech_tokens)[eos_positions])

        # Embed the input sequence
        in_speech_embedding = self.embedding(in_speech_tokens)  # (batch_size, speech_max_len + 1, d_model)
        in_speech_embedding_lens = speech_tokens_lens + 1
        input_lens = llm_hidden_states.size(1) + speech_max_len + 1
        input_mask = torch.zeros(batch_size, input_lens, input_lens, dtype=torch.bool, device=in_speech_embedding.device)
        not_streaming_flag = []
        for i in range(batch_size):
            not_streaming_flag.append(1)
            # attn v1: speech emb可以看到完整的text emb
            input_mask[i, :llm_hidden_lens[i], :llm_hidden_lens[i]] = True
            input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + in_speech_embedding_lens[i], llm_hidden_states.size(1): llm_hidden_states.size(1) + in_speech_embedding_lens[i]] = subsequent_mask(in_speech_embedding_lens[i], in_speech_embedding.device)
            input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + in_speech_embedding_lens[i], :llm_hidden_lens[i]] = True
           
        result = self.model.forward(
            input_ids=input_ids,
            attention_mask=input_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )
        return result.loss


    def generate_units(self, tts_inputs, new_hidden_states, new_tokens, is_finished=False):
        # only for batch size = 1
        new_hidden_states = self.input_proj(new_hidden_states)
        new_token_embeddings = self.model.get_input_embeddings()(new_tokens)
        new_hidden_states = self.fusion(new_hidden_states, new_token_embeddings)
        if tts_inputs is not None:
            tts_inputs = torch.cat([tts_inputs, new_hidden_states], dim=0)
        else:
            tts_inputs = new_hidden_states
        if is_finished:
            device = tts_inputs.device
            sep_id = torch.LongTensor([self.tokenizer.convert_tokens_to_ids("<sep>")]).to(device)
            sep_emb = self.model.get_input_embeddings()(sep_id)
            tts_inputs = torch.cat([tts_inputs, sep_emb], dim=0)

        _, M = eval(self.stream_params)
        max_new_tokens = M if not is_finished else 1024
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=tts_inputs.unsqueeze(0),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0]
        generated_units = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_tokens_embeds = self.model.get_input_embeddings()(generated_tokens)
        tts_inputs = torch.cat([tts_inputs, generated_tokens_embeds], dim=0)
        return tts_inputs, generated_units

class LLMSpeechGenerator(nn.Module):
    def __init__(self, config):
        super(LLMSpeechGenerator, self).__init__()
        self.model = Qwen2ForCausalLM(Qwen2Config(**config.speech_generator))
        self.tokenizer = AutoTokenizer.from_pretrained(config.tts_tokenizer)
        self.input_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, self.model.config.hidden_size)
        )
        self.stream_params = config.stream_params
        self.gate = nn.Sequential(
            nn.Linear(2 * self.model.config.hidden_size, self.model.config.hidden_size),
        nn.Sigmoid()
        )
        self.adaln = AdaLN(config.hidden_size, 768)
        
        # Define special tokens
        self.bos_token = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
        self.eos_token = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        self.padding_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        # Assuming sos_token is same as bos_token or define it separately
        self.sos_token = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1

    def fusion(self, rep, emb):
        gate = self.gate(torch.cat([rep, emb], dim=-1))
        return rep * gate + emb * (1 - gate)
    
    def prepare_inputs_labels_for_speech_and_text(
        self, 
        llm_hidden_states,  # List of hidden states from LLM
        llm_labels,         # List of labels corresponding to LLM tokens
        speech_tokens,      # Target speech tokens to generate
        emotion_features=None,  # Optional emotion features
        txt_eos_emb=None,      # Optional text EOS embedding
        past_key_values=None
    ):
        """
        Prepare inputs and labels for speech generation from LLM hidden states.
        """
        batch_size = len(llm_hidden_states)
        device = llm_hidden_states[0].device
        
        # Process LLM hidden states - filter out ignored tokens
        filtered_llm_hidden_list = []
        llm_hidden_lens = []
        
        for i, (llm_rep, llm_label) in enumerate(zip(llm_hidden_states, llm_labels)):
            # Filter out IGNORE_INDEX and special token 128009
            valid_mask = torch.logical_and(llm_label != IGNORE_INDEX, llm_label != 128009)
            llm_hidden_filter = llm_rep[valid_mask]
            
            # Apply AdaLN if emotion features are provided
            if emotion_features is not None and hasattr(self, 'adaln'):
                emotion_feat = emotion_features[i] if isinstance(emotion_features, list) else emotion_features[i]
                # Apply AdaLN to all but last token
                if llm_hidden_filter.shape[0] > 1:
                    llm_hidden_filter[:-1] = self.adaln(llm_hidden_filter[:-1], emotion_feat)
            
            # Replace last token with txt_eos_emb if provided
            if txt_eos_emb is not None and llm_hidden_filter.shape[0] > 0:
                if txt_eos_emb.dim() == 3:  # (1, 1, hidden_size)
                    txt_eos_emb = txt_eos_emb.squeeze(0)  # (1, hidden_size)
                if txt_eos_emb.dim() == 2 and txt_eos_emb.shape[0] == 1:
                    txt_eos_emb = txt_eos_emb.squeeze(0)  # (hidden_size,)
                llm_hidden_filter = torch.cat([llm_hidden_filter[:-1], txt_eos_emb.unsqueeze(0)], dim=0)
            
            filtered_llm_hidden_list.append(llm_hidden_filter)
            llm_hidden_lens.append(llm_hidden_filter.shape[0])
        
        # Project LLM hidden states to model dimension
        projected_llm_hidden_list = []
        for hidden in filtered_llm_hidden_list:
            projected = self.input_proj(hidden)
            projected_llm_hidden_list.append(projected)
        
        # Process speech tokens
        speech_tokens = speech_tokens.clone()
        speech_tokens[speech_tokens == IGNORE_INDEX] = self.padding_token
        
        # Calculate actual speech lengths
        speech_lens = []
        for tokens in speech_tokens:
            actual_len = (tokens != self.padding_token).sum().item()
            speech_lens.append(actual_len)
        
        # Prepare special tokens
        bos_tokens = torch.full((batch_size, 1), self.bos_token, dtype=torch.long, device=device)
        sos_tokens = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=device)
        eos_tokens = torch.full((batch_size, 1), self.eos_token, dtype=torch.long, device=device)
        
        # Get embeddings
        embed_layer = self.model.get_input_embeddings()
        bos_embeds = embed_layer(bos_tokens)  # (batch_size, 1, embed_dim)
        sos_embeds = embed_layer(sos_tokens)  # (batch_size, 1, embed_dim)
        
        # Create input speech tokens with SOS at beginning
        input_speech_tokens = torch.cat([sos_tokens, speech_tokens], dim=1)
        input_speech_embeds = embed_layer(input_speech_tokens)
        
        # Create output speech tokens with EOS at actual end
        output_speech_tokens = speech_tokens.clone()
        for i in range(batch_size):
            if speech_lens[i] < output_speech_tokens.shape[1]:
                output_speech_tokens[i, speech_lens[i]] = self.eos_token
                # Set everything after EOS to padding
                if speech_lens[i] + 1 < output_speech_tokens.shape[1]:
                    output_speech_tokens[i, speech_lens[i] + 1:] = self.padding_token
        
        # Combine embeddings
        new_input_embeds = []
        new_labels = []
        
        for i in range(batch_size):
            # LLM part: BOS + projected hidden states
            llm_part = torch.cat([bos_embeds[i], projected_llm_hidden_list[i]], dim=0)
            
            # Speech part: SOS + speech tokens
            speech_part = input_speech_embeds[i, :speech_lens[i] + 1]  # +1 for SOS
            
            # Combine
            combined_embeds = torch.cat([llm_part, speech_part], dim=0)
            new_input_embeds.append(combined_embeds)
            
            # Create labels
            # No loss on LLM part and input speech tokens
            llm_labels_part = torch.full((llm_part.shape[0],), IGNORE_INDEX, dtype=torch.long, device=device)
            
            # Labels for speech generation (shifted by 1 due to SOS)
            speech_labels_part = output_speech_tokens[i, :speech_lens[i] + 1]  # +1 to include EOS
            
            combined_labels = torch.cat([llm_labels_part, speech_labels_part], dim=0)
            new_labels.append(combined_labels)
        
        # Pad to max length
        max_len = max(embeds.shape[0] for embeds in new_input_embeds)
        
        input_embeds_padded = torch.zeros(batch_size, max_len, new_input_embeds[0].shape[1], 
                                         dtype=new_input_embeds[0].dtype, device=device)
        labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        
        # Fill padded tensors
        for i in range(batch_size):
            cur_len = new_input_embeds[i].shape[0]
            input_embeds_padded[i, :cur_len] = new_input_embeds[i]
            labels_padded[i, :cur_len] = new_labels[i]
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=device)
        
        # Shift labels for next token prediction
        # The model predicts token i+1 from token i
        labels_shifted = labels_padded[:, 1:].contiguous()
        # Pad the last position
        labels_final = torch.cat([
            labels_shifted,
            torch.full((batch_size, 1), IGNORE_INDEX, dtype=torch.long, device=device)
        ], dim=1)
        
        return input_embeds_padded, attention_mask, position_ids, labels_final, past_key_values
