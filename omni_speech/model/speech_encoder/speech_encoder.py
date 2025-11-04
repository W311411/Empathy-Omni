# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import json
import torch
import torch.nn as nn
import numpy as np
import whisper
import pdb

from safetensors.torch import load_file
from omni_speech.model.speech_tokenizer.modeling_whisper import WhisperVQEncoder,WhisperVQConfig

import sys
import os

from funasr import AutoModel
from typing import Tuple
from dataclasses import dataclass

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)


        from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

        whisper = pipeline("automatic-speech-recognition", model_config.speech_encoder, torch_dtype=torch.bfloat16, device="cpu",chunk_length_s=30, batch_size=256)
        encoder = whisper.model.get_encoder()
        
        replace_layer_norm(encoder)
        return encoder

class WhisperModelLoader:
    @classmethod
    def load(cls, model_config):
        # whisper_model = WhisperForConditionalGeneration.from_pretrained(model_config.speech_encoder).model.encoder.to(torch.bfloat16).to('cpu')
        
        Whisper_config = WhisperVQConfig.from_pretrained(model_config.speech_encoder)
        whisper_model = WhisperVQEncoder(config=Whisper_config).eval().to('cpu')
        pretrained_state_dict = load_file(model_config.speech_encoder + '/model.safetensors')
        whisper_model.load_state_dict(pretrained_state_dict)
        whisper_model.to(torch.bfloat16)
        
        return whisper_model

@dataclass
class UserDirModule:
    user_dir: str

class Emotion2vecEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        model_path = UserDirModule(model_config.encoder_fairseq_dir)
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.emotion_encoder])
        model = model[0]

        return model

from omni_speech.model.speech_encoder.sensevoice_model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr import AutoModel
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

class SenseVoiceEncoder(SenseVoiceSmall):
    def __init__(self):  # 修正: 语法错误，应该是 def __init__(self):
        super().__init__()  # 修正: 语法错误，应该是 super().__init__()
    
    def encode_features(
        self,
        data_in,
        data_lengths=None,
        frontend=None,
        **kwargs,
    ):
        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            speech, speech_lengths = extract_fbank(
                data_in, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        language = kwargs.get("language", "auto")
        language_query = self.embed(
            torch.LongTensor(
                [[self.lid_dict[language] if language in self.lid_dict else 0]]
            ).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        
        use_itn = kwargs.get("use_itn", False)
        output_timestamp = kwargs.get("output_timestamp", False)

        textnorm = kwargs.get("text_norm", None)
        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # Encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        return encoder_out, encoder_out_lens