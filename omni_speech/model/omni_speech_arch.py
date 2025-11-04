# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
#    Copyright 2023 Haotian Liu
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

from abc import ABC, abstractmethod
import pdb

import torch

from .speech_encoder.builder import build_speech_encoder, build_emotion_encoder
from .speech_projector.builder import build_speech_projector, build_emotion_projector, build_emotion_predict_head, build_emotion_classifier
from .speech_generator.builder import build_speech_generator
from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from omni_speech.utils import lengths_to_padding_mask
from transformers import WhisperFeatureExtractor
import torch.nn.functional as F

class OmniSpeechMetaModel:

    def __init__(self, config):
        super(OmniSpeechMetaModel, self).__init__(config)
        speech_encoder_type = getattr(config, 'speech_encoder_type', None)
        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)
        if hasattr(config, "emotion_encoder"):
            # self.emotion_encoder = build_emotion_encoder(config)
            self.emotion_projector = build_emotion_projector(config)
            self.emotion_predict_head = build_emotion_predict_head(config)    
        if speech_encoder_type and any(encoder_type in speech_encoder_type.lower() for encoder_type in ['glm4voice', 'whisper_stream']):
            self.feature_extractor = build_feature_extractor(config)
        else:
            self.feature_extractor = None
        if getattr(config, 'use_duplex', False):
            self.duplex_predictor = build_duplex_predictor(config)
        # if hasattr(config, "speech_generator_type"):
        #     self.speech_generator = build_speech_generator(config)

    def get_speech_encoder(self):
        speech_encoder = getattr(self, 'speech_encoder', None)
        if type(speech_encoder) is list:
            speech_encoder = speech_encoder[0]
        return speech_encoder
    
    def get_emotion_encoder(self):
        emotion_encoder = getattr(self, 'emotion_encoder', None)
        if type(emotion_encoder) is list:
            emotion_encoder = emotion_encoder[0]
        return emotion_encoder

    def get_feature_extractor(self):
        return getattr(self, 'feature_extractor', None) 

    def initialize_speech_modules(self, model_args, fsdp=None):       
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.emotion_encoder = getattr(model_args, "emotion_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.use_duplex = getattr(model_args, 'use_duplex', False)
        self.config.fullduplex_nhead = getattr(model_args, 'fullduplex_nhead', 8)
        self.config.fullduplex_dropout = getattr(model_args, 'fullduplex_dropout', 0.1)
        self.config.fullduplex_num_classes = getattr(model_args, 'fullduplex_num_classes', 3)
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)
        self.config.emotion_encoder_ds_rate = getattr(model_args, 'emotion_encoder_ds_rate', 5),
        self.config.emotion_encoder_hidden_size = getattr(model_args, 'emotion_encoder_hidden_size', 768)
        self.config.deepspeed_config = getattr(model_args, 'deepspeed_config', None)


        if self.get_speech_encoder() is None:
            speech_encoder = build_speech_encoder(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.speech_encoder = [speech_encoder]
            else:
                self.speech_encoder = speech_encoder  
                
        if self.config.emotion_encoder is not None:
            emotion_encoder = build_emotion_encoder(model_args)

            self.emotion_encoder = emotion_encoder

        if getattr(self, 'feature_extractor', None) is None:
            if self.config.speech_encoder_type in ['glm4voice', 'whisper_stream']:
                self.feature_extractor = build_feature_extractor(self.config)
            
        if getattr(self, 'speech_projector', None) is None:
            self.speech_projector = build_speech_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.speech_projector.parameters():
                p.requires_grad = True
    
        if getattr(self, 'emotion_projector', None) is None:
            self.emotion_projector = build_emotion_projector(self.config)
            self.emotion_predict_head = build_emotion_predict_head(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.emotion_projector.parameters():
                p.requires_grad = True
       

        # # code for the speech generator
        # self.config.speech_generator_type = getattr(model_args, 'speech_generator_type', 'ctc')
        # self.config.ctc_decoder_config = getattr(model_args, 'ctc_decoder_config', '(4,4096,32,11008)')
        # self.config.ctc_upsample_factor = getattr(model_args, 'ctc_upsample_factor', 1)
        # self.config.ctc_loss_weight = getattr(model_args, 'ctc_loss_weight', 1.0)
        # self.config.unit_vocab_size = getattr(model_args, 'unit_vocab_size', 1000)
        # # self.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', False)
        # if getattr(self, "speech_generator", None) is None:
        #     self.speech_generator = build_speech_generator(self.config)


class OmniSpeechMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()
    
    def get_emotion_encoder(self):
        return self.get_model().get_emotion_encoder()
    
    def get_speech_projector(self):
        return self.get_model().speech_projector
    
    def get_emotion_projector(self):
        return self.get_model().emotion_projector
    
    def get_duplex_predictor(self):
        return self.get_model().duplex_predictor
    
    def get_feature_extractor(self):
        return self.get_model().feature_extractor
    
    def get_emotion_predict_head(self):
        return self.get_model().emotion_predict_head
    
    def encode_in_emotion(self, wav, wav_lengths):
        emotion_encoder = self.get_emotion_encoder()
        if emotion_encoder is None:
            return None 
        # import pdb; pdb.set_trace() 
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        encoder_outs = emotion_encoder.extract_features(wav, None)['x']
        emotion_projector = self.get_emotion_projector()
        encoder_outs = emotion_projector(encoder_outs)
        wav_lengths = wav_lengths // emotion_projector.k
        emotion_features = [encoder_outs[i, :wav_lengths[i]] for i in range(len(encoder_outs))]
        return emotion_features
    
    
    def encode_out_emotion(self, wav):
        """
        编码音频并根据wav_lengths平均特征
        Args:
            wav: 音频输入 [batch_size, seq_len] 或 [seq_len]
            wav_lengths: 音频长度列表
        Returns:
            平均后的emotion features [batch_size, feature_dim]
        """
        emotion_encoder = self.get_emotion_encoder()
        if emotion_encoder is None:
            return None

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        

        encoder_outs = emotion_encoder.extract_features(wav, None)['x'].mean(dim=1)
        
        return encoder_outs

    def encode_speech(self, speech, speech_lengths):
        speech_encoder_type = self.config.speech_encoder_type
        speech_encoder = self.get_speech_encoder()
        feature_extractor = self.get_feature_extractor()

        if "whisper" == speech_encoder_type.lower():
            encoder_outs = speech_encoder(speech.permute(0, 2, 1))['last_hidden_state'] # torch.Size([1, 1500, 1280])
            # encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            speech_lengths = (speech_lengths + 1) // 2
        elif "glm4voice" in speech_encoder_type.lower():
            encoder_outs, speech_lengths = extract_speech_features(speech_encoder, feature_extractor, speech)
        elif "whisper_stream" in speech_encoder_type.lower():
            encoder_outs, speech_lengths = stream_extract_speech_features(speech_encoder, feature_extractor, speech)
        elif "sensevoice_small" in speech_encoder_type.lower():
            # encoder_outs, speech_lengths, _ , _ = speech_encoder.inference_long_audio(speech)
            _, encoder_outs, speech_lengths = speech_encoder.generate(
                input=speech,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
        elif "cosyvoice2" in speech_encoder_type.lower():
            encoder_outs, _ = speech_encoder.extract_speech_token(speech)
        else:
            raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')

        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        
        # if self.config.use_duplex:
        #     duplex_predictor = self.get_duplex_predictor()
        if speech_projector_type == "linear":
            encoder_outs = speech_projector(encoder_outs) # GLM4VOICE torch.Size([1, 12, 4096])
        else:
            raise ValueError(f'Unknown speech projector: {speech_projector_type}')

        if any(keyword in speech_encoder_type.lower() for keyword in ['whisper', 'glm4voice', 'whisper_stream', 'sensevoice_small']):
            speech_lengths = speech_lengths // speech_projector.k
            speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        else:
            speech_features = encoder_outs
              
        return speech_features

    
    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths, wav, wav_lengths, emotion_wavs
    ):
        import time
        total_start_time = time.time()
        # 1. 初始检查和获取语音编码器
        check_start_time = time.time()
        speech_encoder = self.get_speech_encoder()
        # import pdb; pdb.set_trace()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
            check_time = (time.time() - check_start_time) * 1000
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None
        check_time = (time.time() - check_start_time) * 1000
        # 2. 编码语音
        encode_start_time = time.time()
        # import pdb; pdb.set_trace()
        speech_features = self.encode_speech(speech, speech_lengths)
        emotion_features = self.encode_in_emotion(wav, wav_lengths)
        encode_time = (time.time() - encode_start_time) * 1000

        # 3. 处理输入参数（填充 None）
        prep_start_time = time.time()
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        prep_time = (time.time() - prep_start_time) * 1000

        # 4. 根据注意力掩码移除填充
        mask_start_time = time.time()
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        mask_time = (time.time() - mask_start_time) * 1000

        # 5. 处理输入嵌入和标签
        embed_start_time = time.time()
        new_input_embeds = []
        new_labels = []
        cur_speech_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            # import pdb; pdb.set_trace() 
            if num_speech == 0:
                # cur_speech_features = speech_features[cur_speech_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # cur_input_embeds = torch.cat([cur_input_embeds_1, cur_speech_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds_1)
                new_labels.append(labels[batch_idx])
                cur_speech_idx += 1
                continue

            speech_token_indices = [-1] + torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_nospeech = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech = []
            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_nospeech.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_nospeech.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nospeech]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospeech))
            cur_input_embeds_no_speech = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_speech + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_nospeech[i])
                if i < num_speech:
                    if emotion_features is not None:
                        speech_feat = speech_features[cur_speech_idx]
                        emotion_feat = emotion_features[cur_speech_idx]                       
                        speech_len = speech_feat.shape[0]
                        emotion_len = emotion_feat.shape[0]
                        
                        if emotion_len != speech_len:
                            if emotion_len > speech_len:
                                indices = torch.linspace(0, emotion_len - 1, speech_len).long()
                                emotion_feat = emotion_feat[indices]
                            else:
                                padding_needed = speech_len - emotion_len
                                emotion_feat = F.pad(emotion_feat, (0, 0, 0, padding_needed), value=0)
                        
                        # 创建交错特征
                        cur_speech_features = torch.zeros(2 * speech_len, speech_feat.shape[1],
                                                        device=speech_feat.device, dtype=speech_feat.dtype)
                        cur_speech_features[::2] = speech_feat
                        cur_speech_features[1::2] = emotion_feat
                    else:
                        cur_speech_features = speech_features[cur_speech_idx]
                    cur_speech_idx += 1
                    cur_new_input_embeds.append(cur_speech_features)
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        embed_time = (time.time() - embed_start_time) * 1000

        # 6. 截断序列到最大长度
        truncate_start_time = time.time()
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        truncate_time = (time.time() - truncate_start_time) * 1000

        # 7. 组合和填充
        combine_start_time = time.time()
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask_padded = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids_padded = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask_padded[i, -cur_len:] = True
                    position_ids_padded[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask_padded[i, :cur_len] = True
                    position_ids_padded[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask_padded.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if emotion_wavs is None:
            print('1111111111')
            out_emotion_features = None
        else:
            out_emotion_features = self.encode_out_emotion(emotion_wavs)
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, out_emotion_features
        

    
