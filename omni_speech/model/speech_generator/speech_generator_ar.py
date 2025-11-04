import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig, LlamaRMSNorm, LlamaRotaryEmbedding
from omni_speech.constants import IGNORE_INDEX
import copy
from transformers.cache_utils import DynamicCache
from omni_speech.model.mask import *
import pdb

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

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return ~mask

def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, shift_already: bool = False, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    if shift_already:
        shift_logits = logits
        shift_labels = labels
    else:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    # pdb.set_trace()
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index=-1):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index)
        
    def forward(self, logits, target):
        """
        logits: B*T1*D
        target: B*T2
        """
        logits = logits.transpose(1, 2)
        target = target.to(torch.long)
        loss = self.criterion(logits, target)
        return loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=8192):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_table = torch.zeros(max_len, d_model)
        pos_table[:, 0::2] = torch.sin(position * div_term)
        pos_table[:, 1::2] = torch.cos(position * div_term)
        self.pos_table = pos_table.unsqueeze(0)

    def forward(self, enc_inputs):
        try:
            enc_inputs = enc_inputs + self.pos_table[:, :enc_inputs.size(1), :].to(enc_inputs.device)
        except Exception as e:
            print(f"Error: {e}")
            print(f"enc_inputs shape: {enc_inputs.shape}")
            print(f"pos_table shape: {self.pos_table[:, :enc_inputs.size(1), :].shape}")
            print(f"enc_inputs device: {enc_inputs.device}")
            exit(1)
        return self.dropout(enc_inputs)



class SpeechGeneratorAR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_length = config.max_seq_length
        self.hidden_size = config.llm_hidden_size  
        self.speech_vocab_size = config.unit_vocab_size + config.special_tokens 
        self.max_speech_tokens = config.max_speech_tokens
        self.bos_token = config.speech_bos_token_id
        self.sos_token = config.speech_sos_token_id
        self.eos_token = config.speech_eos_token_id
        self.padding_token = config.speech_padding_token_id

        llama_config = LlamaConfig(
            vocab_size=self.speech_vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=config.decoder_ffn_dim,
            num_hidden_layers=config.decoder_num_layers,
            num_attention_heads=config.decoder_num_heads,
            max_position_embeddings=config.speech_max_position_embeddings,
            bos_token_id=self.bos_token,
            eos_token_id=self.eos_token,
            pad_token_id=self.padding_token,
            attention_dropout=config.decoder_dropout
        )

        self.embedding = nn.Embedding(
            self.speech_vocab_size,
            self.hidden_size,
            padding_idx=self.padding_token
        )
        
        self.gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )
        
        self.encode_layers = nn.ModuleList(
            [LlamaDecoderLayer(llama_config, layer_idx) for layer_idx in range(config.encoder_num_layers)]
        )
        self.norm = LlamaRMSNorm(llama_config.hidden_size)
        self.encode_rotary_emb = LlamaRotaryEmbedding(config=llama_config)

        self.decode_layers = nn.ModuleList(
            [LlamaDecoderLayer(llama_config, layer_idx) for layer_idx in range(config.decoder_num_layers)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=llama_config)

        self.dropout = nn.Dropout(p=config.decoder_dropout)
        self.output_proj = nn.Linear(self.hidden_size, self.speech_vocab_size)
        self.criterion = CrossEntropyLoss(ignore_index=self.padding_token)
        self.txt_token_num = getattr(config, 'txt_token_num', 5)
        self.speech_token_num = getattr(config, 'speech_token_num', 15)


    def pre_nn_forward(self, hidden, hidden_lens):
        inputs_embeds = hidden
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + \
                                      inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.encode_rotary_emb(hidden_states, position_ids)
        batch_size, max_len, _ = hidden.size()
        input_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=hidden.device)
        for i in range(batch_size):
            # input_mask[i, :hidden_lens[i], :hidden_lens[i]] = True
            input_mask[i, :hidden_lens[i], :hidden_lens[i]] = subsequent_mask(hidden_lens[i], hidden.device)
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(inputs_embeds.dtype).min
        for decoder_layer in self.encode_layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
        return hidden_states

    def forward(self, tgt_reps, labels, tgt_units):
        # # # start preparing
        batch_size = len(tgt_reps)
        tgt_label_reps = []
        hidden_reps_lens = []
        for tgt_rep, label in zip(tgt_reps, labels):
            tgt_label_reps.append(tgt_rep[label != IGNORE_INDEX])
            hidden_reps_lens.append(tgt_label_reps[-1].shape[0])
        hidden_reps_lens = torch.tensor(hidden_reps_lens).to(tgt_label_reps[0].device)

        # pdb.set_trace()
        max_len = max([rep.size(0) for rep in tgt_label_reps])
        hidden_states = torch.zeros(len(tgt_label_reps), max_len, tgt_label_reps[0].size(1), device=tgt_label_reps[0].device, dtype=tgt_reps[0].dtype)
        for i, rep in enumerate(tgt_label_reps):
            hidden_states[i, :rep.size(0), :] = rep

        past_key_values = DynamicCache.from_legacy_cache(None)

        bos_token = torch.full((batch_size, 1), self.bos_token, dtype=torch.long, device=hidden_states.device)
        sos_token = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=hidden_states.device)
        eos_token = torch.full((batch_size, 1), self.eos_token, dtype=torch.long, device=hidden_states.device)
        padding_token = torch.full((batch_size, 1), self.padding_token, dtype=torch.long, device=hidden_states.device)

        tgt_units[tgt_units == IGNORE_INDEX] = self.padding_token
        tgt_units_lens = []
        for tgt_unit in tgt_units:
            tgt_units_lens.append(torch.sum(tgt_unit != self.padding_token))
        tgt_units_lens = torch.tensor(tgt_units_lens).to(tgt_label_reps[0].device)

        # # # start forwarding
        # pdb.set_trace()
        hidden_states = self.pre_nn_forward(hidden_states, hidden_reps_lens)
        bos_emb = self.embedding(bos_token)
        hidden_states = torch.cat([bos_emb, hidden_states], dim=1)
        hidden_reps_lens = hidden_reps_lens + 1

        # Create input x with sos token at the beginning
        speech_max_len = tgt_units.shape[1]
        in_tgt_units = torch.cat([sos_token, tgt_units], dim=1) 
        
        # Create output y with eos token at the end
        out_tgt_units = torch.cat([tgt_units, padding_token], dim=1)
        eos_positions = torch.arange(speech_max_len + 1, device=tgt_units.device).expand(batch_size, speech_max_len + 1) == tgt_units_lens.unsqueeze(1)
        out_tgt_units = out_tgt_units.masked_scatter(eos_positions, eos_token.expand_as(out_tgt_units)[eos_positions])

        # Embed the input sequence
        in_tgt_reps = self.embedding(in_tgt_units)  # (batch_size, speech_max_len + 1, d_model)
        in_tgt_reps_lens = tgt_units_lens + 1
        input_lens = hidden_states.size(1) + speech_max_len + 1
        input_mask = torch.zeros(batch_size, input_lens, input_lens, dtype=torch.bool, device=in_tgt_reps.device)
        for i in range(batch_size):
            if torch.rand(1).item() > 0.5:
                # attn v1: speech emb可以看到完整的text emb
                input_mask[i, :hidden_reps_lens[i], :hidden_reps_lens[i]] = True
                input_mask[i, hidden_states.size(1): hidden_states.size(1) + in_tgt_reps_lens[i], hidden_states.size(1): hidden_states.size(1) + in_tgt_reps_lens[i]] = subsequent_mask(in_tgt_reps_lens[i], in_tgt_reps.device)
                input_mask[i, hidden_states.size(1): hidden_states.size(1) + in_tgt_reps_lens[i], :hidden_reps_lens[i]] = True
            else:
                # attn v3: streaming mask
                input_mask[i, :hidden_reps_lens[i], :hidden_reps_lens[i]] = subsequent_mask(hidden_reps_lens[i], hidden_states.device)
                input_mask[i, hidden_states.size(1): hidden_states.size(1) + in_tgt_reps_lens[i], hidden_states.size(1): hidden_states.size(1) + in_tgt_reps_lens[i]] = subsequent_mask(in_tgt_reps_lens[i], in_tgt_reps.device)

                # 修改后的第三部分：分块处理输入到目标的注意力掩码
                sp_start = hidden_states.size(1)
                sp_len = in_tgt_reps_lens[i]
                num_chunks = (sp_len + self.speech_token_num - 2) // self.speech_token_num
                for k in range(num_chunks):
                    # 目标块在序列中的位置
                    chunk_start = k * self.speech_token_num + 1
                    chunk_end = min((k + 1) * self.speech_token_num + 1, sp_len)
                    if chunk_start == 1:
                        chunk_start -= 1
                    tgt_slice = slice(sp_start + chunk_start, sp_start + chunk_end)
                    
                    # 计算当前块可见的输入范围
                    visible_limit = (k + 1) * self.txt_token_num  # 能看到前k+1个输入块
                    visible_limit = min(visible_limit, hidden_reps_lens[i])  # 不超过实际输入长度
                    if chunk_end == sp_len and visible_limit != hidden_reps_lens[i]:
                        print(hidden_reps_lens[i], in_tgt_reps_lens[i])
                        raise ValueError("Invalid chunk end")
                    
                    # 设置该块的注意力掩码
                    input_mask[i, tgt_slice, :visible_limit] = True

        # Pass through the transformer
        inputs_embeds = torch.cat([hidden_states, in_tgt_reps], 1)
        hidden_states = self.dropout(hidden_states)
        llm_hidden_len = hidden_states.size(1)
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(inputs_embeds.dtype).min
        for decoder_layer in self.decode_layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)

        encoder_out = hidden_states[:, llm_hidden_len:]

        # Project to vocabulary size
        logits = self.output_proj(encoder_out)
        loss = ForCausalLMLoss(logits, out_tgt_units, vocab_size=self.speech_vocab_size, ignore_index=self.padding_token, shift_already=True)
        # loss = self.criterion(logits, out_tgt_units)

        # Return logits and loss (if available) in a dictionary
        result = {'logits': logits}
        if loss is not None:
            result['loss'] = loss

        return loss


    # def predict(self, hidden, top_k, prefix, penalty_window_size, penalty, max_tokens=1000):
    def predict(self, hidden, top_k=1, prefix=None, penalty_window_size=0, penalty=0, max_tokens=1000):
        # Pass through pre_nn
        hidden = self.pre_nn_forward(hidden, [hidden.size(1)])
        # Concat bos embedding
        bos_emb = self.embedding(torch.full((1, 1), self.bos_token, dtype=torch.long, device=hidden.device))
        hidden = torch.cat([bos_emb, hidden], dim=1)
        # init past key values
        past_key_values = DynamicCache.from_legacy_cache(None)

        inputs_embeds = hidden
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
                                      device=inputs_embeds.device)
        hidden_states = self.transformer_infer_v2(inputs_embeds, cache_position, past_key_values)

        # init generated tokens
        cur_token = torch.full((1, 1), self.sos_token, dtype=torch.long, device=hidden.device)
        generated_tokens = torch.full((1, 1), self.sos_token, dtype=torch.long, device=hidden.device)
        # generate tokens
        # pdb.set_trace()
        for i in range(max_tokens):
            inputs_embeds = self.embedding(cur_token)
            past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            hidden_states = self.transformer_infer_v2(inputs_embeds, cache_position, past_key_values)
            hidden_states = self.norm(hidden_states)

            # Project to vocabulary size
            logits = self.output_proj(hidden_states)

            # apply penalty
            if penalty_window_size > 0:
                for token in set(generated_tokens[0][-penalty_window_size:]):
                    logits[:, :, token] /= penalty

            # top k sampling
            output = logits.squeeze(0).squeeze(0)
            probs = torch.nn.functional.softmax(output, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
            probs = probs / probs.sum()
            next_token_id = torch.multinomial(probs, 1).unsqueeze(0)

            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
            cur_token = next_token_id

            # eos
            if next_token_id == self.eos_token:
                break
        # pdb.set_trace()
        return generated_tokens


    def transformer_infer_v2(self, inputs_embeds, cache_position, past_key_values):
        position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        next_decoder_cache = None

        max_len = inputs_embeds.shape[1] + past_key_values.get_seq_length()
        input_mask = torch.zeros(inputs_embeds.shape[0], max_len, max_len, dtype=torch.bool, device=inputs_embeds.device)
        input_mask[0, :max_len, :max_len] = subsequent_mask(max_len, inputs_embeds.device)
        input_mask = input_mask[:, -inputs_embeds.shape[1]:, :]
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(inputs_embeds.dtype).min

        for decoder_layer in self.decode_layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
        return hidden_states
    

    # def transformer_infer(self, inputs_embeds, cache_position, past_key_values):
    #     position_ids = cache_position.unsqueeze(0)
    #     hidden_states = inputs_embeds
    #     position_embeddings = self.rotary_emb(hidden_states, position_ids)
    #     next_decoder_cache = None
    #     for decoder_layer in self.decode_layers:
    #         layer_outputs = decoder_layer(
    #             hidden_states,
    #             attention_mask=None,
    #             position_ids=position_ids,
    #             past_key_value=past_key_values,
    #             output_attentions=False,
    #             use_cache=True,
    #             cache_position=None,
    #             position_embeddings=position_embeddings,
    #         )
    #         hidden_states = layer_outputs[0]
    #         next_decoder_cache = layer_outputs[1]
    #     return hidden_states
            



