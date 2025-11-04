import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig, LlamaRMSNorm, LlamaRotaryEmbedding
from omni_speech.constants import IGNORE_INDEX
import copy
from transformers.cache_utils import DynamicCache
from omni_speech.model.mask import *
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from datetime import datetime
from scipy.stats import entropy

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

class AdaLN(nn.Module):
    def __init__(self, normalized_shape, condition_dim, eps=1e-5):
        """
        AdaLN: Adaptive Layer Normalization.
        
        Args:
            normalized_shape (int): The shape of the input to normalize.
            condition_dim (int): The dimensionality of the condition vector (e.g., emotion features).
            eps (float, optional): A small value to prevent division by zero in normalization. Default is 1e-5.
        """
        super(AdaLN, self).__init__()
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Projection layers for condition to adaptively control the normalization
        self.condition_projection = nn.Linear(condition_dim, 2 * normalized_shape)
        
        # Initialize weights and biases
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)
        nn.init.ones_(self.condition_projection.bias[:normalized_shape])

    def forward(self, x, condition):
        """
        Forward pass for AdaLN.
        
        Args:
            x (Tensor): The input tensor to be normalized, of shape [batch_size, seq_len, hidden_size].
            condition (Tensor): The condition tensor used to adapt the normalization, of shape [batch_size, condition_dim].
        
        Returns:
            Tensor: The normalized tensor, with the same shape as input `x`.
        """
        # Check if input x is 2D or 3D
        original_shape = x.shape
        is_2d_input = len(original_shape) == 2
        
        # If it's a 2D input, we add a batch dimension
        if is_2d_input:
            x = x.unsqueeze(0)
        
        # Calculate the mean and variance for normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize the input
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        
        # Project the condition vector to obtain adaptive scale and shift
        gamma_beta = self.condition_projection(condition)
        
        # Split the projected values into gamma (scale) and beta (shift)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        
        # Apply the adaptive scaling and shifting
        out = gamma * x_hat + beta
        
        # If the input was 2D, we remove the added batch dimension
        if is_2d_input:
            out = out.squeeze(0)
        
        return out
    
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

class SpeechGeneratorARMTP(nn.Module):
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
        self.speech_generator_stage = None
        self.mtp_num = config.mtp_num
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

        self.gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )

        self.embedding = nn.Embedding(
            self.speech_vocab_size,
            self.hidden_size,
            padding_idx=self.padding_token
        )
        # nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
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
        self.adaln = AdaLN(self.hidden_size, 768)
        self.mtp_layers = nn.ModuleList()
        for _ in range(self.mtp_num):
            # 'linear': nn.Linear(self.hidden_size*2, self.hidden_size),
            mtp_layer = nn.ModuleDict({
            'rotary_emb': LlamaRotaryEmbedding(config=llama_config),
            'decoder_layer': LlamaDecoderLayer(llama_config, layer_idx=0),
            'norm_final': LlamaRMSNorm(llama_config.hidden_size),
            'output_proj': nn.Linear(self.hidden_size, self.speech_vocab_size),
            })
            self.mtp_layers.append(mtp_layer)
        self.txt_token_num = getattr(config, 'txt_token_num', 5)
        self.speech_token_num = getattr(config, 'speech_token_num', 15)
        self.reset_streaming_cache()

    def fusion(self, rep, emb):
        gate = self.gate(torch.cat([rep, emb], dim=-1))
        # import pdb; pdb.set_trace()
        return rep * gate + emb * (1 - gate)

    def forward_mtp_layer(self, hidden_state, mtp_layer, input_mask, llm_hidden_len):
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(hidden_state.dtype).min
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + \
                                      hidden_state.shape[1], device=hidden_state.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = mtp_layer['rotary_emb'](hidden_state, position_ids)
        hidden_state = mtp_layer['decoder_layer'](hidden_state, attention_mask=attention_mask, position_ids=position_ids, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=position_embeddings)[0]
        output = mtp_layer['norm_final'](hidden_state)
        logits = mtp_layer['output_proj'](output[:, llm_hidden_len:])
        # logits = logits
        
        return hidden_state, logits

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


    def forward(self, llm_hidden_list, llm_labels, text_embedding_list, speech_tokens, txt_eos_emb=None, emotion_list=None):
        # print(speech_tokens.shape)
        too_long_flag = True
        batch_size = len(llm_hidden_list)

        self.speech_generator_stage = 3
        processed_hidden_states = []
        hidden_state_lengths = []

        for hidden_rep, label, text_emb, emotion in zip(llm_hidden_list, llm_labels, text_embedding_list, emotion_list):
            # 过滤有效tokens
            valid_token_mask = torch.logical_and(label != IGNORE_INDEX, label != 128009)
            filtered_hidden = hidden_rep[valid_token_mask]
            # 根据阶段处理
            if self.speech_generator_stage == 2:
                processed_rep = text_emb
            elif self.speech_generator_stage == 3:
                processed_rep = self.fusion(filtered_hidden, text_emb)
            else:
                processed_rep = filtered_hidden
            
            # 应用AdaLN和EOS处理
            if txt_eos_emb is not None:
                final_hidden = processed_rep
                # 对除最后一个token外应用adaln
                if emotion.dim() == 1:
                    emotion_expanded = emotion.unsqueeze(0).expand(processed_rep.shape[0]-1, -1)
                else:
                    emotion_expanded = emotion
                adaln_output = self.adaln(processed_rep[:-1, :], emotion_expanded)
                # 拼接EOS嵌入
                final_hidden = torch.cat([adaln_output, txt_eos_emb.squeeze(0)], dim=0)
            else:
                final_hidden = processed_rep
            
            processed_hidden_states.append(final_hidden)
            hidden_state_lengths.append(final_hidden.shape[0])
        llm_hidden_lens = torch.tensor(hidden_state_lengths).to(processed_hidden_states[0].device)

        max_len = max([rep.size(0) for rep in processed_hidden_states])
        llm_hidden_states = torch.zeros(len(processed_hidden_states), max_len, processed_hidden_states[0].size(1), device=processed_hidden_states[0].device, dtype=processed_hidden_states[0].dtype)
        for i, rep in enumerate(processed_hidden_states):
            llm_hidden_states[i, :rep.size(0), :] = rep

        past_key_values = DynamicCache.from_legacy_cache(None)

        bos_token = torch.full((batch_size, 1), self.bos_token, dtype=torch.long, device=llm_hidden_states.device)
        sos_token = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=llm_hidden_states.device)
        eos_token = torch.full((batch_size, 1), self.eos_token, dtype=torch.long, device=llm_hidden_states.device)
        padding_token = torch.full((batch_size, 1), self.padding_token, dtype=torch.long, device=llm_hidden_states.device)

        speech_tokens[speech_tokens == IGNORE_INDEX] = self.padding_token
        speech_tokens_lens = []
        for tgt_unit in speech_tokens:
            speech_tokens_lens.append(torch.sum(tgt_unit != self.padding_token))
        speech_tokens_lens = torch.tensor(speech_tokens_lens).to(processed_hidden_states[0].device)

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
            if torch.rand(1).item() > 0.5:
                not_streaming_flag.append(1)
                # attn v1: speech emb可以看到完整的text emb
                input_mask[i, :llm_hidden_lens[i], :llm_hidden_lens[i]] = True
                input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + in_speech_embedding_lens[i], llm_hidden_states.size(1): llm_hidden_states.size(1) + in_speech_embedding_lens[i]] = subsequent_mask(in_speech_embedding_lens[i], in_speech_embedding.device)
                input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + in_speech_embedding_lens[i], :llm_hidden_lens[i]] = True
            else:
                not_streaming_flag.append(0)
                # attn v3: streaming mask
                input_mask[i, :llm_hidden_lens[i], :llm_hidden_lens[i]] = subsequent_mask(llm_hidden_lens[i], llm_hidden_states.device)
                input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + in_speech_embedding_lens[i], llm_hidden_states.size(1): llm_hidden_states.size(1) + in_speech_embedding_lens[i]] = subsequent_mask(in_speech_embedding_lens[i], in_speech_embedding.device)

                # 修改后的第三部分：分块处理输入到目标的注意力掩码
                sp_start = llm_hidden_states.size(1)
                sp_len = in_speech_embedding_lens[i]
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
                    visible_limit = min(visible_limit, llm_hidden_lens[i])  # 不超过实际输入长度
                    if chunk_end == sp_len and (visible_limit != llm_hidden_lens[i]):
                        if (llm_hidden_lens[i] - visible_limit < 2):
                            pass
                        elif too_long_flag:
                            pass
                        else:
                            print(llm_hidden_lens[i], in_speech_embedding_lens[i])
                            raise ValueError("Invalid chunk end")
                    
                    # 设置该块的注意力掩码
                    input_mask[i, tgt_slice, :visible_limit] = True

        # # Pass through the transformer
        # llm_hidden_states = self.dropout(llm_hidden_states)
        llm_hidden_len = llm_hidden_states.size(1)
        hidden_states = torch.cat([llm_hidden_states, in_speech_embedding], 1)

        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(hidden_states.dtype).min

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
        norm_hidden_states = self.norm(hidden_states)

        encoder_out = norm_hidden_states[:, llm_hidden_len:]

        # Project to vocabulary size
        logits = self.output_proj(encoder_out)
        loss = ForCausalLMLoss(logits, out_speech_tokens, vocab_size=self.speech_vocab_size, ignore_index=self.padding_token, shift_already=True)

        # mtp_loss computation
        # mtp_loss = 0
        # last_mtp_loss = 0
        # lastmtp_speech_token = in_speech_tokens
        # lastmtp_speech_label = out_speech_tokens
        # speech_reps_lens = in_speech_embedding_lens.clone()
        # last_hidden_states = hidden_states
        # factor = 0.8
        # mtp_loss_list = [loss.cpu().detach().numpy()-0]
        # for mtp_k in range(self.mtp_num):
        #     currentmtp_speech_token = torch.cat([lastmtp_speech_token[:,1:], padding_token], dim=1) 
        #     currentmtp_speech_label  = torch.cat([lastmtp_speech_label[:,1:], padding_token], dim=1) 

        #     currentmtp_speech_embedding = self.embedding(currentmtp_speech_token)
        #     speech_reps_lens -= 1
        #     mpt_input_mask = torch.zeros(batch_size, input_lens, input_lens, dtype=torch.bool, device=last_hidden_states.device)
            
        #     for i in range(batch_size):
        #         if not_streaming_flag[i]:
        #             # attn v1: speech emb可以看到完整的text emb
        #             mpt_input_mask[i, :llm_hidden_lens[i], :llm_hidden_lens[i]] = True
        #             mpt_input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + speech_reps_lens[i], llm_hidden_states.size(1): llm_hidden_states.size(1) + speech_reps_lens[i]] = subsequent_mask(speech_reps_lens[i], currentmtp_speech_embedding.device)
        #             mpt_input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + speech_reps_lens[i], :llm_hidden_lens[i]] = True
        #         else:
        #             mpt_input_mask[i, :llm_hidden_lens[i], :llm_hidden_lens[i]] = subsequent_mask(llm_hidden_lens[i], llm_hidden_states.device)
        #             mpt_input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + speech_reps_lens[i], llm_hidden_states.size(1): llm_hidden_states.size(1) + speech_reps_lens[i]] = subsequent_mask(speech_reps_lens[i], currentmtp_speech_embedding.device)
                    
        #             mpt_input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + speech_reps_lens[i], :llm_hidden_lens[i]] = input_mask[i, llm_hidden_states.size(1) + (in_speech_embedding_lens[i]-speech_reps_lens[i]): llm_hidden_states.size(1) + in_speech_embedding_lens[i], :llm_hidden_lens[i]]
        #             # mpt_input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + speech_reps_lens[i], :llm_hidden_lens[i]] = input_mask[i, llm_hidden_states.size(1):llm_hidden_states.size(1) + speech_reps_lens[i], :llm_hidden_lens[i]]

        #         # print(mpt_input_mask[i, llm_hidden_states.size(1): llm_hidden_states.size(1) + speech_reps_lens[i], :llm_hidden_lens[i]].shape, input_mask[i, llm_hidden_states.size(1) + (in_speech_embedding_lens[i]-speech_reps_lens[i]): llm_hidden_states.size(1) + in_speech_embedding_lens[i], :llm_hidden_lens[i]].shape)
        #         # plt.figure(figsize=(10, 8)) 
        #         # sns.heatmap(mpt_input_mask[i].cpu().numpy(), cmap="viridis", cbar=True)
        #         # plt.title(f"Attention Mask for Batch {i}")
        #         # plt.xlabel("Sequence Length")
        #         # plt.ylabel("Sequence Length")
        #         # plt.savefig('/root/img_tmp.png')
        #         # plt.close()
        #         # pdb.set_trace()

        #     # current_hidden_states = torch.cat([llm_hidden_states, currentmtp_speech_embedding], 1)
        #     current_hidden_states, current_logits = self.forward_mtp_layer(last_hidden_states, self.mtp_layers[mtp_k], mpt_input_mask, llm_hidden_len)
            
        #     mtp_loss += factor * ForCausalLMLoss(current_logits, currentmtp_speech_label, vocab_size=self.speech_vocab_size, ignore_index=self.padding_token, shift_already=True)
        #     mtp_loss_list.append((mtp_loss.cpu().detach().numpy() - last_mtp_loss)/factor)
        #     factor *= 0.8
        #     last_mtp_loss = mtp_loss.cpu().detach().numpy()


        #     last_hidden_states = current_hidden_states
        #     lastmtp_speech_token = currentmtp_speech_token
        #     lastmtp_speech_label = currentmtp_speech_label

        # if torch.rand(1).item() < 0.1:
        #     print(f"mtp:{mtp_loss}, decoder backbone loss:{loss}")
        # loss += mtp_loss

        return loss



    # def predict(self, hidden, top_k, prefix, penalty_window_size, penalty, max_tokens=1000):
    def predict(self, hidden, top_k=2, prefix=None, penalty_window_size=50, penalty=1.2, max_tokens=1024, emotion=None):
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
        hidden_states = self.transformer_infer(inputs_embeds, cache_position, past_key_values)

        # init generated tokens
        cur_token = torch.full((1, 1), self.sos_token, dtype=torch.long, device=hidden.device)
        generated_tokens = torch.full((1, 1), self.sos_token, dtype=torch.long, device=hidden.device)
        # generate tokens


        for i in range(max_tokens):
            inputs_embeds = self.embedding(cur_token)
            past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            hidden_states = self.transformer_infer(inputs_embeds, cache_position, past_key_values)
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
            # pdb.set_trace()

            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
            cur_token = next_token_id

            if next_token_id == self.eos_token:
                break
        # pdb.set_trace()
        # print(generated_tokens)
        return generated_tokens
    

    def transformer_infer(self, inputs_embeds, cache_position, past_key_values):
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
    
    
    def infer_mtp_layer(self, mtp_layer, hidden_state, cache_position, past_key_values):
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = mtp_layer['rotary_emb'](hidden_state, position_ids)

        max_len = hidden_state.shape[1] + past_key_values.get_seq_length()
        input_mask = torch.zeros(hidden_state.shape[0], max_len, max_len, dtype=torch.bool, device=hidden_state.device)
        input_mask[0, :max_len, :max_len] = subsequent_mask(max_len, hidden_state.device)
        input_mask = input_mask[:, -hidden_state.shape[1]:, :]
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(hidden_state.dtype).min

        hidden_state = mtp_layer['decoder_layer'](hidden_state, 
                                            attention_mask=attention_mask, 
                                            position_ids=position_ids, 
                                            past_key_value=past_key_values, 
                                            output_attentions=False, 
                                            use_cache=True, 
                                            cache_position=None, 
                                            position_embeddings=position_embeddings)[0]
        output = mtp_layer['norm_final'](hidden_state)
        logits = mtp_layer['output_proj'](output)
        return hidden_state, logits


    def infer_pre_nn(self, llm_hidden, prenn_cache_position, prenn_past_key_values):
        position_ids = prenn_cache_position.unsqueeze(0)
        hidden_states = llm_hidden
        position_embeddings = self.encode_rotary_emb(hidden_states, position_ids)

        max_len = llm_hidden.shape[1] + prenn_past_key_values.get_seq_length()
        # max_len = llm_hidden.shape[1]
        input_mask = torch.zeros(llm_hidden.shape[0], max_len, max_len, dtype=torch.bool, device=llm_hidden.device)
        input_mask[0, :max_len, :max_len] = subsequent_mask(max_len, llm_hidden.device)
        input_mask = input_mask[:, -llm_hidden.shape[1]:, :]
        # input_mask = input_mask[:, :, :llm_hidden.shape[1]]
        attention_mask = ~(input_mask.unsqueeze(1)) * torch.finfo(llm_hidden.dtype).min

        for encoder_layer in self.encode_layers:
            layer_outputs = encoder_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=prenn_past_key_values, output_attentions=False, use_cache=True, cache_position=None, position_embeddings=position_embeddings)
            hidden_states = layer_outputs[0]
        return hidden_states
    
            

    def predict_mtp(self, llm_hidden, top_k=1, prefix=None, penalty_window_size=5, penalty=2, max_tokens=2048, infer_mtp_token_num=3, emotion=None):
        if infer_mtp_token_num > self.mtp_num:
            raise ValueError("mtp_token_num should be less than mtp_num")
        
        # Pass through pre_nn
        # import pdb; pdb.set_trace()
        llm_hidden = self.pre_nn_forward(llm_hidden, [llm_hidden.size(1)])
        bos_emb = self.embedding(torch.full((1, 1), self.bos_token, dtype=torch.long, device=llm_hidden.device))
        llm_hidden = torch.cat([bos_emb, llm_hidden], dim=1)

        # init past key values
        past_key_values = DynamicCache.from_legacy_cache(None)
        inputs_embeds = llm_hidden
        past_seen_tokens = 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
                                      device=inputs_embeds.device)
        llm_hidden_states = self.transformer_infer(inputs_embeds, cache_position, past_key_values)

        # init mtp past key values
        mtp_past_key_values = []
        for i in range(infer_mtp_token_num):
            mtp_past_key_values.append(DynamicCache.from_legacy_cache(None))
            inputs_embeds = llm_hidden
            past_seen_tokens = 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], \
                                        device=inputs_embeds.device)
            _, _ = self.infer_mtp_layer(self.mtp_layers[i], llm_hidden_states, cache_position, mtp_past_key_values[i])    

        # init generated tokens
        cur_chunk_token = torch.full((1, 1), self.sos_token, dtype=torch.long, device=inputs_embeds.device)
        generated_tokens = torch.full((1, 1), self.sos_token, dtype=torch.long, device=inputs_embeds.device)
        # generate tokens
        while generated_tokens.shape[1] < max_tokens:
        # for i in range(max_tokens):
            inputs_embeds = self.embedding(cur_chunk_token)
            past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            hidden_states = self.transformer_infer(inputs_embeds, cache_position, past_key_values)
            norm_hidden_states = self.norm(hidden_states)

            # Project to vocabulary size
            logits = self.output_proj(norm_hidden_states[:,-1,:].unsqueeze(1))

            # apply penalty
            if penalty_window_size > 0:
                for token in set(generated_tokens[0][-penalty_window_size:]):
                    logits[:, :, token] /= penalty

            # top k sampling
            output = logits.squeeze(0).squeeze(0)
            probs = torch.nn.functional.softmax(output, dim=-1)
            try:
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
                probs = probs / probs.sum()
                next_token_id = torch.multinomial(probs, 1).unsqueeze(0)
            except:
                pdb.set_trace()

            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
            cur_chunk_token = torch.cat([cur_chunk_token, next_token_id], dim=-1)
            cur_chunk_token = cur_chunk_token[:,1:]
            # single_cur_token = next_token_id
            # pre_chunk_token = cur_chunk_token
            # cur_chunk_token = single_cur_token
            if next_token_id == self.eos_token:
                break
            
            # 第一次infer，每个mtp模块都只需要接受1个输入，所以单独拎出来写
            if generated_tokens.shape[-1] <= infer_mtp_token_num + 2:
                for j in range(infer_mtp_token_num):
                    mtp_inputs_embeds = self.embedding(next_token_id)
                    past_seen_tokens = mtp_past_key_values[j].get_seq_length()
                    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + mtp_inputs_embeds.shape[1], device=mtp_inputs_embeds.device)
                    hidden_states, logits = self.infer_mtp_layer(self.mtp_layers[j], hidden_states, cache_position, mtp_past_key_values[j])

                    output = logits.squeeze(0).squeeze(0)
                    probs = torch.nn.functional.softmax(output, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
                    probs = probs / probs.sum()
                    next_token_id = torch.multinomial(probs, 1).unsqueeze(0)

                    generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
                    cur_chunk_token = torch.cat([cur_chunk_token, next_token_id], dim=-1)
                    if next_token_id == self.eos_token:
                        break
                if next_token_id == self.eos_token:
                    break
            else:
                for j in range(infer_mtp_token_num):
                    mtp_inputs_embeds = self.embedding(cur_chunk_token)
                    past_seen_tokens = mtp_past_key_values[j].get_seq_length()
                    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + mtp_inputs_embeds.shape[1], device=mtp_inputs_embeds.device)
                    hidden_states, logits = self.infer_mtp_layer(self.mtp_layers[j], hidden_states, cache_position, mtp_past_key_values[j])

                    logits = logits[:,-1,:]
                    output = logits.squeeze(0).squeeze(0)
                    probs = torch.nn.functional.softmax(output, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
                    probs = probs / probs.sum()
                    next_token_id = torch.multinomial(probs, 1).unsqueeze(0)

                    generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
                    cur_chunk_token = torch.cat([cur_chunk_token, next_token_id], dim=-1)
                    cur_chunk_token = cur_chunk_token[:,1:]
                    if next_token_id == self.eos_token:
                        break
                if next_token_id == self.eos_token:
                    break
        return generated_tokens
    
    
    def merge_caches(self, cache1, cache2):
        if cache1.get_seq_length() == 0:
            return copy.deepcopy(cache2)
        if cache2.get_seq_length() == 0:
            return copy.deepcopy(cache1)

        legacy_cache1 = cache1.to_legacy_cache()
        legacy_cache2 = cache2.to_legacy_cache()
        merged_cache = []

        for layer_cache1, layer_cache2 in zip(legacy_cache1, legacy_cache2):
            merged_layer_cache = []
            for tensor1, tensor2 in zip(layer_cache1, layer_cache2):
                merged_tensor = torch.cat((tensor1.clone(), tensor2.clone()), dim=2)
                merged_layer_cache.append(merged_tensor)
            merged_cache.append(tuple(merged_layer_cache))

        return DynamicCache.from_legacy_cache(tuple(merged_cache))
    
    def split_cache(self, cache, split_idx):
        legacy_cache = cache.to_legacy_cache()
        cache1 = []
        cache2 = []

        for layer_cache in legacy_cache:
            layer_cache1 = []
            layer_cache2 = []
            for tensor in layer_cache:
                tensor1 = tensor[:, :, :split_idx, :].clone()
                tensor2 = tensor[:, :, split_idx:, :].clone()
                layer_cache1.append(tensor1)
                layer_cache2.append(tensor2)
            cache1.append(tuple(layer_cache1))
            cache2.append(tuple(layer_cache2))

        return DynamicCache.from_legacy_cache(tuple(cache1)), DynamicCache.from_legacy_cache(tuple(cache2))
    
    def streaming_predict_mtp(self, llm_hidden, top_k=1, prefix=None, penalty_window_size=6, penalty=2, max_tokens=2048, infer_mtp_token_num=3, emotion=None):        
        if infer_mtp_token_num > self.mtp_num:
            raise ValueError("mtp_token_num should be less than mtp_num")
        llm_hidden = self.adaln(llm_hidden, emotion)
        first_call = self._all_generated_tokens is None
        if first_call:
            self._prenn_past_key_values = DynamicCache.from_legacy_cache(None)
            self._speech_decoder_past_key_values = {
                "text": DynamicCache.from_legacy_cache(None), 
                "speech": DynamicCache.from_legacy_cache(None)
            }
            self._mtp_past_key_values = []
            for i in range(infer_mtp_token_num):
                self._mtp_past_key_values.append(copy.deepcopy(self._speech_decoder_past_key_values))
            try:
                self._all_generated_tokens = torch.full((1, 1), self.sos_token, dtype=torch.long, device=llm_hidden.device)
            except:
                self._all_generated_tokens = torch.full((1, 1), self.sos_token, dtype=torch.long, device=self._speech_decoder_past_key_values['text'][0][0].device)
        
        if llm_hidden is not None:
            prenn_past_seen_tokens = self._prenn_past_key_values.get_seq_length()
            prenn_cache_position = torch.arange(prenn_past_seen_tokens, prenn_past_seen_tokens + llm_hidden.shape[1], device=llm_hidden.device)
            llm_hidden = self.infer_pre_nn(llm_hidden, prenn_cache_position, self._prenn_past_key_values)
        
        if self._speech_decoder_past_key_values['text'].get_seq_length() == 0:
            bos_emb = self.embedding(torch.full((1, 1), self.bos_token, dtype=torch.long, device=llm_hidden.device))
            llm_hidden = torch.cat([bos_emb, llm_hidden], dim=1)
            cur_chunk_token = torch.full((1, 1), self.sos_token, dtype=torch.long, device=llm_hidden.device)
            generated_tokens = torch.empty((1, 0), dtype=torch.long, device=llm_hidden.device)
        else:
            cur_chunk_token = self._all_generated_tokens[:,-1-infer_mtp_token_num:]
            try:
                generated_tokens = torch.empty((1, 0), dtype=torch.long, device=self._speech_decoder_past_key_values['text'][0][0].device)
            except:
                generated_tokens = torch.empty((1, 0), dtype=torch.long, device=llm_hidden.device)

        if llm_hidden is not None: 
            # print(cur_chunk_token.shape, self._all_generated_tokens.shape)
            speech_decoder_past_seen_tokens_text = self._speech_decoder_past_key_values['text'].get_seq_length()
            speech_decoder_cache_position = torch.arange(speech_decoder_past_seen_tokens_text, speech_decoder_past_seen_tokens_text + llm_hidden.shape[1], device=llm_hidden.device)

            llm_hidden_states_passsd = self.transformer_infer(llm_hidden, speech_decoder_cache_position, self._speech_decoder_past_key_values['text'])
            _, previous_speech_past_key_values = self.split_cache(self._speech_decoder_past_key_values['speech'], speech_decoder_past_seen_tokens_text)
            self._speech_decoder_past_key_values['speech'] = self.merge_caches(copy.deepcopy(self._speech_decoder_past_key_values['text']), previous_speech_past_key_values)
            # self._speech_decoder_past_key_values['speech'] = copy.deepcopy(self._speech_decoder_past_key_values['text'])
            
            for i in range(infer_mtp_token_num):
                mtp_past_seen_tokens_text = self._mtp_past_key_values[i]['text'].get_seq_length()
                cache_position = torch.arange(mtp_past_seen_tokens_text, mtp_past_seen_tokens_text + llm_hidden.shape[1], device=llm_hidden.device)

                _, _ = self.infer_mtp_layer(self.mtp_layers[i], llm_hidden_states_passsd, cache_position, self._mtp_past_key_values[i]['text'])
                _, previous_mpt_speech_past_key_values = self.split_cache(self._mtp_past_key_values[i]['speech'], mtp_past_seen_tokens_text)
                self._mtp_past_key_values[i]['speech'] = self.merge_caches(copy.deepcopy(self._mtp_past_key_values[i]['text']), previous_mpt_speech_past_key_values)
                # self._mtp_past_key_values[i]['speech'] = copy.deepcopy(self._mtp_past_key_values[i]['text'])

        if self._is_last_chunk:
            cycle_times = math.ceil((max_tokens - self._all_generated_tokens.shape[-1]) / (infer_mtp_token_num + 1))
        else:
            cycle_times = math.ceil(self.speech_token_num / (infer_mtp_token_num + 1))

        for _ in range(cycle_times):
            inputs_embeds = self.embedding(cur_chunk_token)
            speech_decoder_past_seen_tokens_all = self._speech_decoder_past_key_values['speech'].get_seq_length()
            cache_position = torch.arange(speech_decoder_past_seen_tokens_all, speech_decoder_past_seen_tokens_all + inputs_embeds.shape[1], device=inputs_embeds.device)
            hidden_states = self.transformer_infer(inputs_embeds, cache_position, self._speech_decoder_past_key_values['speech'])
            norm_hidden_states = self.norm(hidden_states)
            logits = self.output_proj(norm_hidden_states[:,-1,:].unsqueeze(1))
            
            for token in set(generated_tokens[0][-penalty_window_size:]):
                logits[:, :, token] /= penalty

            output = logits.squeeze(0).squeeze(0)
            probs = torch.nn.functional.softmax(output, dim=-1)
            
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
            probs = probs / probs.sum()
            next_token_id = torch.multinomial(probs, 1).unsqueeze(0)

            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
            cur_chunk_token = torch.cat([cur_chunk_token, next_token_id], dim=-1)
            cur_chunk_token = cur_chunk_token[:,1:]
            
            if next_token_id == self.eos_token:
                break
                
            if generated_tokens.shape[-1] >= self.speech_token_num and cycle_times == math.ceil(self.speech_token_num / (infer_mtp_token_num + 1)):
                break
            
            if self._all_generated_tokens.shape[1] == 1 and generated_tokens.shape[1] < infer_mtp_token_num + 1:
                for j in range(infer_mtp_token_num):
                    mtp_inputs_embeds = self.embedding(next_token_id)
                    past_seen_tokens = self._mtp_past_key_values[j]['speech'].get_seq_length()
                    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + mtp_inputs_embeds.shape[1], device=mtp_inputs_embeds.device)
                    hidden_states, logits = self.infer_mtp_layer(self.mtp_layers[j], hidden_states, cache_position, self._mtp_past_key_values[j]['speech'])
                    for token in set(generated_tokens[0][-penalty_window_size:] if generated_tokens.shape[1] > 0 else []):
                        logits[:, :, token] /= penalty
                    output = logits.squeeze(0).squeeze(0)
                    probs = torch.nn.functional.softmax(output, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
                    probs = probs / probs.sum()
                    next_token_id = torch.multinomial(probs, 1).unsqueeze(0)

                    generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
                    cur_chunk_token = torch.cat([cur_chunk_token, next_token_id], dim=-1)
                    if next_token_id == self.eos_token:
                        break
                    if generated_tokens.shape[-1] >= self.speech_token_num and cycle_times == math.ceil(self.speech_token_num / (infer_mtp_token_num + 1)):
                        break
                if next_token_id == self.eos_token:
                    break
                if generated_tokens.shape[-1] >= self.speech_token_num and cycle_times == math.ceil(self.speech_token_num / (infer_mtp_token_num + 1)):
                    break
            else:
                for j in range(infer_mtp_token_num):
                    mtp_inputs_embeds = self.embedding(cur_chunk_token)
                    past_seen_tokens = self._mtp_past_key_values[j]['speech'].get_seq_length()
                    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + mtp_inputs_embeds.shape[1], device=mtp_inputs_embeds.device)
                    hidden_states, logits = self.infer_mtp_layer(self.mtp_layers[j], hidden_states, cache_position, self._mtp_past_key_values[j]['speech'])
                    logits = logits[:,-1,:].unsqueeze(1)
                    for token in set(generated_tokens[0][-penalty_window_size:] if generated_tokens.shape[1] > 0 else []):
                        logits[:, :, token] /= penalty
                    output = logits.squeeze(0).squeeze(0)
                    probs = torch.nn.functional.softmax(output, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
                    probs = probs / probs.sum()
                    next_token_id = torch.multinomial(probs, 1).unsqueeze(0)

                    generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
                    cur_chunk_token = torch.cat([cur_chunk_token, next_token_id], dim=-1)
                    cur_chunk_token = cur_chunk_token[:,1:]
                    if next_token_id == self.eos_token:
                        break
                    if generated_tokens.shape[-1] >= self.speech_token_num and cycle_times == math.ceil(self.speech_token_num / (infer_mtp_token_num + 1)):
                        break
                if next_token_id == self.eos_token:
                    break
                if generated_tokens.shape[-1] >= self.speech_token_num and cycle_times == math.ceil(self.speech_token_num / (infer_mtp_token_num + 1)):
                    break
                if cur_chunk_token.shape[-1] != infer_mtp_token_num + 1:
                    cur_chunk_token = cur_chunk_token[:, -infer_mtp_token_num-1:]
        
        self._all_generated_tokens = torch.cat([self._all_generated_tokens, generated_tokens], dim=-1)
        
        return generated_tokens
        

    def reset_streaming_cache(self):
        self._prenn_past_key_values = DynamicCache.from_legacy_cache(None)
        self._speech_decoder_past_key_values = {
            "text": DynamicCache.from_legacy_cache(None),
            "speech": DynamicCache.from_legacy_cache(None)
        }
        self._mtp_past_key_values = [copy.deepcopy(self._speech_decoder_past_key_values) for _ in range(self.mtp_num)]
        self._all_generated_tokens = None
        self._is_last_chunk = False
            
    def set_last_chunk(self, is_last=True):
        self._is_last_chunk = is_last
