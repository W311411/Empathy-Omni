import re
from typing import Optional, Union, Tuple, List, Generator
import pdb
import torch
import yaml
import torch.nn as nn
import time
import timeit
from transformers.cache_utils import DynamicCache

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from omni_speech.model.language_model.omni_speech_llama import OmniSpeechLlamaForCausalLM
from omni_speech.model.speech_generator.builder import build_speech_generator
from omni_speech.model.speech_generator.generation import GenerationWithCTC
from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX

SENTENCE_DELIM_RE = re.compile(r'[。：？！.?!\n]$')

class OmniSpeech2SConfig(LlamaConfig):
    model_type = "omni_speech2s_llama"


def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
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


class OmniSpeech2SLlamaForCausalLM(OmniSpeechLlamaForCausalLM, GenerationWithCTC):
    config_class = OmniSpeech2SConfig

    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.tokenizer = tokenizer
        
        # self.enable_streaming = getattr(config, "enable_streaming", False) # streaming
        if hasattr(config, "speech_generator_type"):
            # print(config)
            self.tune_speech_generator_only = False
            self.speech_generator = build_speech_generator(config)
        else:
            # self.speech_generator = None
            config.speech_generator_type = getattr(config, "speech_generator_type", "ar_mtp")
            config.speech_generator_config = getattr(config, "speech_generator_config", "./scripts/mtp/ar_config_5.yaml")
            print("config.speech_generator_config:", config.speech_generator_config)
 
            self.initialize_speech_generator(config)
            # print("speech generator config is:")
            # print(config)
            
        self.reset_streaming_state()
        self.post_init()

    
    def reset_streaming_state(self):
        self.generated_ids = []
        self.past_key_values = None
        self.cur_hidden_states = []
        self.cur_text = ""
        self.units_preds = []
        self.last_id_embeds = None
    
    def initialize_speech_generator(self, model_args):
        self.config.speech_generator_type = model_args.speech_generator_type
        if 'ar' in self.config.speech_generator_type:
            self.config.speech_generator_config = getattr(model_args, 'speech_generator_config')
            with open(self.config.speech_generator_config, 'r') as file:
                arconfig = yaml.safe_load(file)
            
            self.config.llm_hidden_size = arconfig.get('llm_hidden_size', 2048)
            self.config.decoder_hidden_size = arconfig.get('decoder_hidden_size', 2048)
            self.config.decoder_num_heads = arconfig.get('decoder_num_heads', 32)
            self.config.decoder_ffn_dim = arconfig.get('decoder_ffn_dim', 8192)
            self.config.decoder_dropout = arconfig.get('decoder_dropout', 0.1)
            self.config.decoder_num_layers = arconfig.get('decoder_num_layers', 4)
            self.config.encoder_num_layers = arconfig.get('encoder_num_layers', 2)  
            self.config.unit_vocab_size = arconfig.get('unit_vocab_size', 6561)
            self.config.max_speech_tokens = arconfig.get('max_speech_tokens', 4096)
            self.config.max_seq_length = arconfig.get('max_seq_length', 8192)  
            self.config.special_tokens = arconfig.get('special_tokens', 4)
            self.config.speech_bos_token_id = arconfig.get('speech_bos_token_id', self.config.unit_vocab_size + 0)
            self.config.speech_sos_token_id = arconfig.get('speech_sos_token_id', self.config.unit_vocab_size + 1)
            self.config.speech_eos_token_id = arconfig.get('speech_eos_token_id', self.config.unit_vocab_size + 2)
            self.config.speech_padding_token_id = arconfig.get('speech_padding_token_id', self.config.unit_vocab_size + 3)
            self.config.switch_token_id = arconfig.get('switch_token_id', self.config.unit_vocab_size + 4)
            self.config.speech_max_position_embeddings = arconfig.get('speech_max_position_embeddings', 2048)
            self.config.gen_loss_weight = arconfig.get('gen_loss_weight', 1.0)
            self.config.group_size = arconfig.get('group_size', 5)
            self.config.txt_token_num = arconfig.get('txt_token_num', 5)
            self.config.speech_token_num = arconfig.get('speech_token_num', 15)
            self.config.mtp_num = arconfig.get('mtp_num', 5)
            self.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', True)

            self.speech_generator = build_speech_generator(self.config)
        else:
            raise NotImplementedError(f"`{self.config.speech_generator_type}` is not supported in ctc configuration. Please use `--speech_generator_type ctc`")

    def get_speech_decoder(self):  
        return self.speech_generator

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
        tgt_units: Optional[torch.LongTensor] = None,
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
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )
        
        if self.training:
            if self.tune_speech_generator_only:
                with torch.no_grad():
                    llama_output = super(OmniSpeechLlamaForCausalLM, self).forward(
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
                # pdb.set_trace()
                txt_eos_emb = self.get_model().embed_tokens(torch.tensor([[128009]], device=llama_output['hidden_states'][-1].device))
                loss = self.speech_generator(llama_output['hidden_states'][-1], labels, tgt_units, txt_eos_emb)
            else:
                # pdb.set_trace()
                llama_output = super(OmniSpeechLlamaForCausalLM, self).forward(
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
                lm_loss = llama_output.loss
                txt_eos_emb = self.get_model().embed_tokens(torch.tensor([[128009]], device=llama_output['hidden_states'][-1].device))
                sp_loss = self.speech_generator(llama_output['hidden_states'][-1], labels, tgt_units, txt_eos_emb)

                loss = lm_loss + sp_loss * self.config.gen_loss_weight
        else:
            llama_output = super(OmniSpeechLlamaForCausalLM, self).forward(
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
            loss = llama_output.loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=llama_output.logits,
            past_key_values=llama_output.past_key_values,
            hidden_states=llama_output.hidden_states,
            attentions=llama_output.attentions
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        streaming_unit_gen=False,
        infer_mtp_token_num=0,
        streaming=False,
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
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        outputs = GenerationWithCTC.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            streaming_unit_gen=streaming_unit_gen,
            **kwargs
        )

        hidden_states = outputs['hidden_states']
        hidden_states = torch.cat([hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(1, len(hidden_states))], dim=1)

        stop_tokens = [13, 30, 0, 11] # , . ! ?
        text_token = outputs['sequences'][0].tolist()
        stop_token_indices = [i for i, token in enumerate(text_token) if token in stop_tokens] # [0, 59, 77]
        filter_stop_token_indices = [0]
        token_num_bound = 50
        while stop_token_indices:
            index = stop_token_indices.pop(0)
            if index - filter_stop_token_indices[-1] > token_num_bound:
                filter_stop_token_indices.append(index)
        if filter_stop_token_indices[-1] == len(text_token) - 2:
            filter_stop_token_indices[-1] = len(text_token) - 1
        else:
            filter_stop_token_indices += [len(text_token) - 1]
        filter_stop_token_indices = [index + 1 for index in filter_stop_token_indices]
        filter_stop_token_indices[0] = 0
        # if len(filter_stop_token_indices) > 3 and filter_stop_token_indices[-1] - filter_stop_token_indices[-2] < token_num_bound:
        #     filter_stop_token_indices.pop(-2)
        txt_eos_emb = self.get_model().embed_tokens(torch.tensor([[128009]], device=hidden_states.device))
        hidden_states_list = [hidden_states[:, filter_stop_token_indices[i]:filter_stop_token_indices[i+1], :] for i in range(len(filter_stop_token_indices) - 1)]
        hidden_states_list = [torch.cat([hidden[:, :, :], txt_eos_emb], dim=1) for hidden in hidden_states_list]
        segment_seq = [outputs['sequences'][:, filter_stop_token_indices[i]:filter_stop_token_indices[i+1]] for i in range(len(filter_stop_token_indices) - 1)]
        
        units_pred_list = [torch.tensor([[self.config.speech_sos_token_id]], device=hidden_states.device)]
        for hidden in hidden_states_list:
            if infer_mtp_token_num > 0:
                if streaming:
                    units_pred_list.append(self.speech_generator.pseudo_streaming_predict_mtp(hidden, infer_mtp_token_num=infer_mtp_token_num)[:,1:-1])
                else:
                    # print("streaming!!")
                    units_pred_list.append(self.speech_generator.predict_mtp(hidden, infer_mtp_token_num=infer_mtp_token_num)[:,1:-1])
            else:
                units_pred_list.append(self.speech_generator.predict(hidden)[:,1:-1])
        units_pred_list.append(torch.tensor([[self.config.speech_eos_token_id]], device=hidden_states.device))
        units_pred = torch.cat(units_pred_list, dim=1).contiguous()

            
        # return segment_seq, units_pred_list
        return outputs.sequences, units_pred


    @torch.no_grad()
    def streaming_generate_mtp(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        infer_mtp_token_num=0,
        txt_token_num=5,
        speech_token_num=15,
        reset_interval=50,  
        max_len = 512,
        **kwargs,
    ) -> Generator[Tuple[str, Optional[List[torch.Tensor]]], None, None]:
        self.txt_token_num = txt_token_num
        self.speech_generator.speech_token_num = speech_token_num
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("inputs_embeds is not supported")
        first_step = self.past_key_values is None

        self.speech_generator.reset_streaming_cache()
        self.speech_generator.set_last_chunk(is_last=False)
        if first_step:
            if speech is not None:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_speech_and_text(
                    inputs,
                    kwargs.get("position_ids", None),
                    kwargs.get("attention_mask", None),
                    None,
                    None,
                    speech,
                    speech_lengths
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)
            current_attention_mask = attention_mask
        else:
            current_attention_mask = torch.full([1, 1], True, device=self.last_id_embeds.device)
        
        generated_ids_list = []
        all_txt_ids = []
        self.units_preds = []
        punctuation_set = ".!?"
        punct_count = 0  
        last_punct_reset = 0  
        txt_eos_emb = self.get_model().embed_tokens(torch.tensor([[128009]], device=inputs_embeds.device))

        while True:
            if len(all_txt_ids) > max_len:
                last_id = torch.tensor([[128009]], device=inputs_embeds.device)
                return_tts_state = txt_eos_emb
            else:
                last_id, self.past_key_values, return_tts_state = self._generate_one_step(
                    inputs_embeds=inputs_embeds if first_step else self.last_id_embeds,
                    attention_mask=current_attention_mask,
                    past_key_values=self.past_key_values,
                    **kwargs
                )
                all_txt_ids.append(last_id)
            
            punct_count += 1  
            generated_ids_list.append(last_id)

            self.cur_hidden_states.append(return_tts_state)
            concat_ids = torch.cat(generated_ids_list, dim=1)
            self.cur_text = self.tokenizer.decode(concat_ids.squeeze(0), skip_special_tokens=True)

            if self.cur_text and (self.cur_text[-1] in punctuation_set) and (punct_count - last_punct_reset >= reset_interval):
                accumulated_hidden = torch.cat(self.cur_hidden_states, dim=1)

                hidden_for_predict = torch.cat([accumulated_hidden, txt_eos_emb], dim=1)
                self.speech_generator.set_last_chunk(is_last=True)

                units_pred = self.speech_generator.streaming_predict_mtp(
                    hidden_for_predict, infer_mtp_token_num=infer_mtp_token_num
                )
                if units_pred is not None:
                    self.units_preds.append(units_pred)
                
                yield concat_ids, [tensor[:, :-1].clone() for tensor in self.units_preds], False
                self.units_preds = []
                generated_ids_list = []
                self.cur_hidden_states = []
                self.cur_text = ""

                self.speech_generator.reset_streaming_cache()
                self.speech_generator.set_last_chunk(is_last=False)
                last_punct_reset = punct_count  

            if len(generated_ids_list) >= self.txt_token_num:
                self.speech_generator.set_last_chunk(is_last=False)
                accumulated_hidden = torch.cat(self.cur_hidden_states, dim=1)
                hidden_for_predict = accumulated_hidden  
                units_pred = self.speech_generator.streaming_predict_mtp(
                    hidden_for_predict, infer_mtp_token_num=infer_mtp_token_num
                )
                if units_pred is not None:
                    self.units_preds.append(units_pred)
                yield concat_ids, self.units_preds.copy(), False
                self.units_preds = []
                generated_ids_list.clear()
                self.cur_hidden_states = []
                self.cur_text = ""

            if last_id[0][0] == 128009:
                self.speech_generator.set_last_chunk(is_last=True)
                if generated_ids_list and generated_ids_list[0][0][0] != 128009:
                    accumulated_hidden = torch.cat(self.cur_hidden_states, dim=1)
                    hidden_for_predict = torch.cat([accumulated_hidden, txt_eos_emb], dim=1)
                    units_pred = self.speech_generator.streaming_predict_mtp(
                        hidden_for_predict, infer_mtp_token_num=infer_mtp_token_num
                    )
                    if units_pred is not None:
                        self.units_preds.append(units_pred)
                    
                    yield concat_ids, [tensor[:, :-1].clone() for tensor in self.units_preds], True
                elif punct_count - last_punct_reset < 2:
                    yield None, [torch.tensor([[6324, 4137]], device=return_tts_state.device)], True
                else:
                    units_pred = self.speech_generator.streaming_predict_mtp(
                        None, infer_mtp_token_num=infer_mtp_token_num
                    )
                    if units_pred is not None:
                        self.units_preds.append(units_pred)
                    yield None, [tensor[:, :-1].clone() for tensor in self.units_preds], True
                    
                break

            if self.past_key_values is not None:
                self.last_id_embeds = self.get_model().embed_tokens(last_id)
                past_len = self.past_key_values[0][0].size(2)
                current_attention_mask = torch.ones(1, past_len + 1, device=self.last_id_embeds.device)
                first_step = False



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

AutoConfig.register("omni_speech2s_llama", OmniSpeech2SConfig)
AutoModelForCausalLM.register(OmniSpeech2SConfig, OmniSpeech2SLlamaForCausalLM)
