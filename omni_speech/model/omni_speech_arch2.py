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


from abc import ABC, abstractmethod

import torch

from .speech_encoder.builder import build_speech_encoder, build_emotion_encoder
from .speech_projector.builder import build_speech_projector, build_emotion_projector, build_emotion_predict_head
from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX


class Omni2SpeechMetaModel:

    def __init__(self, config):
        super(Omni2SpeechMetaModel, self).__init__(config)
        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)
        if hasattr(config, "emotion_encoder"):
            self.emotion_encoder = build_emotion_encoder(config)
            self.emotion_projector = build_emotion_projector(config)
            self.emotion_predict_head = build_emotion_predict_head(config)

    def get_speech_encoder(self):
        speech_encoder = getattr(self, "speech_encoder", None)
        return speech_encoder
    
    def get_emotion_encoder(self):
        emotion_encoder = getattr(self, 'emotion_encoder', None)
        emotion_encoder = getattr(self, "emotion_encoder", None)
        return emotion_encoder
    
    def get_speech_projector(self):
        speech_projector = getattr(self, "speech_projector", None)
        return speech_projector
    
    def get_emotion_projector(self):
        emotion_projector = getattr(self, "emotion_projector", None)
        return emotion_projector

    def initialize_speech_modules(self, model_args):
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(model_args, "speech_encoder_type", None)
        self.config.speech_projector_type = getattr(model_args, 'speech_projector_type', 'linear')
        self.config.speech_encoder_ds_rate = getattr(model_args, 'speech_encoder_ds_rate', 5)
        self.config.speech_encoder_hidden_size = getattr(model_args, 'speech_encoder_hidden_size', 1280)
        self.config.emotion_encoder_ds_rate = getattr(model_args, 'emotion_encoder_ds_rate', 5),
        self.config.emotion_encoder_hidden_size = getattr(model_args, 'emotion_encoder_hidden_size', 768)
        self.config.deepspeed_config = getattr(model_args, 'deepspeed_config', None)

        if self.get_speech_encoder() is None:
            self.speech_encoder = build_speech_encoder(self.config)
        if self.get_speech_projector() is None:
            self.speech_projector = build_speech_projector(self.config)
        if self.get_emotion_encoder() is None:
            self.emotion_encoder = build_emotion_encoder(self.config)
        if self.get_emotion_projector() is None:
            self.emotion_projector = build_emotion_projector(self.config)
        if self.get_emotion_predict_head() is None:
            self.emotion_predict_head = build_emotion_predict_head(self.config)



class Omni2SpeechMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()

    def get_emotion_encoder(self):
        return self.get_model().get_emotion_encoder()
    
    def get_speech_projector(self):
        return self.get_model().get_speech_projector()

    def get_emotion_projector(self):
        return self.get_model().get_emotion_projector()
    
    def get_emotion_predict_head(self):
        return self.get_model().get_emotion_predict_head()

    def encode_in_emotion(self, wav, wav_lengths):
        emotion_encoder = self.get_emotion_encoder()
        if emotion_encoder is None:
            return None
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        encoder_outs = emotion_encoder.extract_features(wav, None)['x']
        emotion_projector = self.get_emotion_projector()
        encoder_outs = emotion_projector(encoder_outs)
        wav_lengths = wav_lengths // emotion_projector.k
        emotion_features = [encoder_outs[i, :wav_lengths[i]] for i in range(len(encoder_outs))]
        return emotion_features

    def encode_speech(self, speech, speech_lengths):
        speech_encoder_type = self.config.speech_encoder_type
        speech_encoder = self.get_speech_encoder()
        if "whisper" in speech_encoder_type.lower():
            encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            speech_lengths = (speech_lengths + 1) // 2
        else:
            raise ValueError(f"Unknown speech encoder type: {speech_encoder_type}")
        
        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        if speech_projector_type == "linear":
            encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f"Unknown speech projector type: {speech_projector_type}")
        
        speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]
        return speech_features

    def encode_out_emotion(self, wav, wav_lengths):
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
        
        # 提取特征
        encoder_outs = emotion_encoder.extract_features(wav, None)['x']
        # encoder_outs shape: [batch_size, seq_len, feature_dim]
        
        batch_size, _, feature_dim = encoder_outs.shape
        averaged_features = torch.zeros(batch_size, feature_dim, device=encoder_outs.device, dtype=torch.bfloat16)
        
        # 对每个样本按有效长度平均
        for i, length in enumerate(wav_lengths):
            valid_features = encoder_outs[i, :length]  # [valid_len, feature_dim]
            averaged_features[i] = valid_features.mean(dim=0)  # [feature_dim]
        
        return averaged_features
    
    def prepare_inputs_labels_for_speech_and_text(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths, wav, wav_lengths, emotion_wavs, emotion_lengths
    ):
        speech_encoder = self.get_speech_encoder()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        speech_features = self.encode_speech(speech, speech_lengths)
        emotion_features = self.encode_in_emotion(wav, wav_lengths)
        
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

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_speech_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
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
        
        assert cur_speech_idx == len(speech_features)

        # Truncate sequences to max length as speech features can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if emotion_wavs is None:
            out_emotion_features = None
        else:
            out_emotion_features = self.encode_out_emotion(emotion_wavs, emotion_lengths)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, out_emotion_features