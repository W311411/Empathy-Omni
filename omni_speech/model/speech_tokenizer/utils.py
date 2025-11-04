import os
import io
import glob
import math
import tarfile
import pdb
import torch
import torch.nn.functional as F
import torchaudio
import safetensors
from .configuration_whisper import WhisperVQConfig,WhisperStreamConfig
from .modeling_whisper import WhisperVQEncoder, WhisperVQForConditionalGeneration
from .modeling_whisper_streaming import WhisperStreamEncoder, WhisperStreamForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast


def load_quantize_encoder(model_path):
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.encoder."):
                    new_key = key[len("model.encoder."):]
                    if new_key.startswith("layer_norm"):
                        continue
                    if new_key.startswith("layers"):
                        layer_id = int(new_key.split(".")[1])
                        if layer_id >= config.quantize_position:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model


_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


def extract_speech_token(model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, utts):
    with torch.no_grad():
        audios, indices = [], []
        for idx, utt in enumerate(utts):
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            else:
                audio, sample_rate = torchaudio.load(utt)
            audio = audio.cuda()
            if sample_rate != 16000:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    ).to('cuda')
                audio = _resample_buffer[sample_rate](audio)
            # if audio.shape[0] > 1:
            #     audio = audio[:1]
            audio = audio[0]
            audio = audio.cpu().numpy()
            time_step = 0
            while time_step * 16000 < audio.shape[0]:
                audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                audios.append(audio_segment)
                indices.append(idx)
                time_step += 30
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
        all_speech_tokens = [[] for _ in range(len(utts))]
        batch_size = 128
        for start in range(0, len(audios), batch_size):
            features = feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                         return_attention_mask=True, return_tensors="pt", device='cuda',
                                         padding="longest", pad_to_multiple_of=stride)
            features = features.to(device="cuda")
            outputs = model(**features)
            speech_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
            attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]
            assert attention_mask.shape == speech_tokens.shape
            for i in range(len(speech_tokens)):
                idx = indices[start + i]
                speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_speech_tokens[idx].extend(speech_token)
        return all_speech_tokens

def extract_speech_features(model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, speech):
    with torch.no_grad():
        all_speech_features = []  
        speech_lengths = []

        for audio_data in speech:
            audio_data = audio_data.cuda()  
            audio_data = audio_data.cpu().to(torch.float32).numpy()  

            audios = []  
            time_step = 0
            batch_size_audio = 1  

            while time_step * 16000 < audio_data.shape[0]:
                audio_segment = audio_data[time_step * 16000: (time_step + 30) * 16000]
                audios.append(audio_segment)
                time_step += 30

            # pooling_kernel_size = model.config.pooling_kernel_size or 1
            # stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
            stride = model.conv1.stride[0] * model.conv2.stride[0] * feature_extractor.hop_length

            batch_size = 128
            for start in range(0, len(audios), batch_size):
                features = feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                             return_attention_mask=True, return_tensors="pt", device='cuda',
                                             padding="longest", pad_to_multiple_of=stride)
                features = features.to(device="cuda")
                features = features.to(model.dtype)
 
                outputs = model(**features)
                speech_features = outputs.last_hidden_state
                # pdb.set_trace()

                attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
                # attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]
                assert attention_mask.shape == speech_features.shape[:2]

                for i in range(batch_size_audio):
                    valid_features = speech_features[i][attention_mask[i].bool()]
                    all_speech_features.append(valid_features)
                    valid_length = valid_features.size(0)  
                    speech_lengths.append(valid_length)


        max_len = max([f.size(0) for f in all_speech_features])  
        padded_features = []
        for f in all_speech_features:
            pad_size = max_len - f.size(0)
            if pad_size > 0:
                f_padded = F.pad(f, (0, 0, pad_size, 0), value=0)
                padded_features.append(f_padded)
            else:
                padded_features.append(f)

        all_speech_features_tensor = torch.stack(padded_features, dim=0).to(model.dtype)
        # pdb.set_trace()
        
        speech_lengths_tensor = torch.tensor(speech_lengths, device=all_speech_features_tensor.device)
        
        return all_speech_features_tensor, speech_lengths_tensor


def stream_extract_speech_features(model: WhisperStreamEncoder, feature_extractor: WhisperFeatureExtractor, speech):
    with torch.no_grad():
        all_speech_features = []  
        speech_lengths = []

        for audio_data in speech:
            audio_data = audio_data.cuda()  
            audio_data = audio_data.cpu().to(torch.float32).numpy()  

            audios = []  
            time_step = 0
            batch_size_audio = 1  

            while time_step * 16000 < audio_data.shape[0]:
                audio_segment = audio_data[time_step * 16000: (time_step + 30) * 16000]
                audios.append(audio_segment)
                time_step += 30
                
            stride = model.conv1.stride[0] * model.conv2.stride[0] * feature_extractor.hop_length

            batch_size = 128
            for start in range(0, len(audios), batch_size):
                features = feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                             return_attention_mask=True, return_tensors="pt", device='cuda',
                                             padding="longest", pad_to_multiple_of=stride)
                features = features.to(device="cuda")
                features = features.to(model.dtype)
                # pdb.set_trace()
                outputs = model(**features)
                speech_features = outputs.last_hidden_state
                # pdb.set_trace()

                attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
                assert attention_mask.shape == speech_features.shape[:2]

                for i in range(batch_size_audio):
                    valid_features = speech_features[i][attention_mask[i].bool()]
                    all_speech_features.append(valid_features)
                    valid_length = valid_features.size(0)  
                    speech_lengths.append(valid_length)


        max_len = max([f.size(0) for f in all_speech_features])  
        padded_features = []
        for f in all_speech_features:
            pad_size = max_len - f.size(0)
            if pad_size > 0:
                f_padded = F.pad(f, (0, 0, pad_size, 0), value=0)
                padded_features.append(f_padded)
            else:
                padded_features.append(f)

        all_speech_features_tensor = torch.stack(padded_features, dim=0).to(model.dtype)
        # pdb.set_trace()
        
        speech_lengths_tensor = torch.tensor(speech_lengths, device=all_speech_features_tensor.device)
        
        return all_speech_features_tensor, speech_lengths_tensor

