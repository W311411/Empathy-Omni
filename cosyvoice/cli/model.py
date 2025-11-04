# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
import pdb


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        if self.fp16 is True:
            self.llm.half()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        assert self.fp16 is True, "we only provide fp16 jit model, set fp16=True if you want to use jit model"
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_onnx(self, flow_decoder_estimator_model):
        import onnxruntime
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        del self.flow.decoder.estimator
        self.flow.decoder.estimator = onnxruntime.InferenceSession(flow_decoder_estimator_model, sess_options=option, providers=providers)

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        if self.fp16 is True:
            llm_embedding = llm_embedding.half()
        with self.llm_context:
            for i in self.llm.inference(text=text.to(self.device),
                                        text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_text=prompt_text.to(self.device),
                                        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                        prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                        embedding=llm_embedding.to(self.device)):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        tts_mel, flow_cache = self.flow.inference(token=token.to(self.device),
                                                  token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                  prompt_token=prompt_token.to(self.device),
                                                  prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                  prompt_feat=prompt_feat.to(self.device),
                                                  prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                  embedding=embedding.to(self.device),
                                                  flow_cache=self.flow_cache_dict[uuid])
        self.flow_cache_dict[uuid] = flow_cache

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)

    def vc(self, source_speech_token, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = source_speech_token.flatten().tolist(), True
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)


class CosyVoice2Model:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        
        self.token_hop_len = 2 * self.flow.input_frame_rate 
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate # 原来=25
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio 
        # hift cache
        self.mel_cache_len = 8 # 原来=8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.token_offsets = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        self.flow.decoder.fp16 = False
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_onnx(self, flow_decoder_estimator_model):
        import onnxruntime
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        del self.flow.decoder.estimator
        self.flow.decoder.estimator = onnxruntime.InferenceSession(flow_decoder_estimator_model, sess_options=option, providers=providers)

    def load_trt(self, flow_decoder_estimator_model):
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()
        self.flow.decoder.fp16 = True

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context:
            for i in self.llm.inference(text=text.to(self.device),
                                        text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_text=prompt_text.to(self.device),
                                        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                        prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                        embedding=llm_embedding.to(self.device)):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                         token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_token=prompt_token.to(self.device),
                                         prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_feat=prompt_feat.to(self.device),
                                         prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                         embedding=embedding.to(self.device),
                                         finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        if stream is True:
            token_offset = 0
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + self.token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     token_offset=token_offset,
                                                     finalize=False)
                    token_offset += self.token_hop_len
                    yield {'tts_speech': this_tts_speech.cpu()}
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < self.token_hop_len + self.flow.pre_lookahead_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=token_offset,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            import ipdb
            ipdb.set_trace()
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=0,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
    def tts_direct(self, speech_tokens: torch.Tensor, flow_embedding: torch.Tensor, prompt_token: torch.Tensor, prompt_feat: torch.Tensor, stream=False, speed=1.0, chunk_para=[3,3,1], **kwargs):
        """
        使用提供的speech tokens进行语音生成。
        """
        # print("tts_direct")
        this_uuid = str(uuid.uuid1())
        txt_num, sp_num, sp_ratio = chunk_para
        sp_len = int(sp_num * sp_ratio)
        # print(self.token_hop_len)
        with self.lock:
            self.tts_speech_token_dict[this_uuid] = speech_tokens.squeeze().tolist()
            speech_tokens = speech_tokens.squeeze().tolist()
            self.hift_cache_dict[this_uuid] = None
            # if stream:
            #     self.llm_end_dict[this_uuid] = False  
            # else:
            #     self.llm_end_dict[this_uuid] = True # 标记LLM处理已结束
        
        if stream:
            token_offset = 0
            # import pdb; pdb.set_trace()
            required_tokens = self.token_hop_len + self.flow.pre_lookahead_len
            
            while True:
                available_tokens = len(speech_tokens[token_offset:])
                required_tokens = self.token_hop_len + self.flow.pre_lookahead_len
                if available_tokens >= required_tokens:
                    # import pdb; pdb.set_trace()
                    # 确保 token 是 2D
                    this_tts_speech_token = torch.tensor(
                        speech_tokens[: token_offset + required_tokens]
                    ).unsqueeze(dim=0).long().to(self.device)  # 转换为 Long 类型并确保形状为 (1, hop_len + pre_lookahead_len)
                    
                    # 确保 prompt_token 是 3D
                    if prompt_token.dim() == 3 and prompt_token.size(1) == 1:
                        prompt_token = prompt_token.squeeze(1)
                    elif prompt_token.dim() != 2:
                        raise ValueError(f"prompt_token 的维度应为 2D, 但得到 {prompt_token.dim()}D")
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=prompt_token,
                        prompt_feat=prompt_feat,
                        embedding=flow_embedding,
                        uuid=this_uuid,
                        token_offset=token_offset,
                        finalize=False
                    )
                    # token_offset += self.token_hop_len
                    token_offset += self.token_hop_len
                    yield {'tts_speech': this_tts_speech.cpu()}
                if len(speech_tokens[token_offset:]) < required_tokens: # <= -> <
                    break
            # 处理剩余的tokens
            this_tts_speech_token = torch.tensor(speech_tokens).unsqueeze(dim=0).long().to(self.device)
            # import pdb; pdb.set_trace()
            this_tts_speech = self.token2wav(
                    token=this_tts_speech_token,
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    embedding=flow_embedding,
                    uuid=this_uuid,
                    token_offset=token_offset,
                    finalize=True,
                    speed=speed
            )
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # 处理所有tokens
            all_tokens = self.tts_speech_token_dict[this_uuid]
            if not all_tokens:
                import pdb; pdb.set_trace()
                raise ValueError("没有可生成的tokens")
            this_tts_speech_token = torch.tensor(all_tokens).long().unsqueeze(dim=0).to(self.device)  # 确保形状为 (1, token_length)
            
            # 确保 prompt_token 是 2D
            if prompt_token.dim() == 3 and prompt_token.size(1) == 1:
                prompt_token = prompt_token.squeeze(1)
                # logging.debug(f"Adjusted prompt_token shape: {prompt_token.shape}, dim: {prompt_token.dim()}")
            elif prompt_token.dim() != 2:
                raise ValueError(f"prompt_token 的维度应为 2D, 但得到 {prompt_token.dim()}D")
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=flow_embedding,
                uuid=this_uuid,
                token_offset=0,
                finalize=True,
                speed=speed
            )
            yield {'tts_speech': this_tts_speech.cpu()}
        
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            # self.llm_end_dict.pop(this_uuid)
    
    def tts_direct_update(self, speech_tokens: torch.Tensor, flow_embedding: torch.Tensor, prompt_token: torch.Tensor, prompt_feat: torch.Tensor, stream=False, speed=1.0, chunk_para=[3,3,1], uuid=None, is_last_speech_chunk=False, **kwargs):
        if uuid is None:
            uuid = str(uuid.uuid1())

        with self.lock:
            if uuid not in self.hift_cache_dict:
                self.hift_cache_dict[uuid] = None
            if uuid not in self.tts_speech_token_dict:
                self.tts_speech_token_dict[uuid] = []
            if uuid not in self.token_offsets:
                self.token_offsets[uuid] = 0
            self.tts_speech_token_dict[uuid].extend(speech_tokens.squeeze().tolist())
            speech_tokens = self.tts_speech_token_dict[uuid]
        
        if stream:
            token_offset = self.token_offsets[uuid]
            # required_tokens = self.token_hop_len + self.flow.pre_lookahead_len
            
            if not is_last_speech_chunk:
                required_tokens = 15 # NOTE
                available_tokens = len(speech_tokens[token_offset:])
                if available_tokens >= required_tokens:
                    this_tts_speech_token = torch.tensor(
                        speech_tokens[: token_offset + required_tokens]
                    ).unsqueeze(dim=0).long().to(self.device)
                    
                    if prompt_token.dim() == 3 and prompt_token.size(1) == 1:
                        prompt_token = prompt_token.squeeze(1)
                    elif prompt_token.dim() != 2:
                        raise ValueError(f"prompt_token 的维度应为 2D, 但得到 {prompt_token.dim()}D")
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=prompt_token,
                        prompt_feat=prompt_feat,
                        embedding=flow_embedding,
                        uuid=uuid,
                        token_offset=token_offset,
                        finalize=False
                    )
                    # token_offset += self.token_hop_len
                    token_offset += 12 # NOTE
                    self.token_offsets[uuid] = token_offset
                    yield {'tts_speech': this_tts_speech.cpu()}
            else:
                while True:
                    required_tokens = self.token_hop_len + self.flow.pre_lookahead_len # NOTE
                    available_tokens = len(speech_tokens[token_offset:])
                    if available_tokens >= required_tokens:
                        this_tts_speech_token = torch.tensor(
                            speech_tokens[: token_offset + required_tokens]
                        ).unsqueeze(dim=0).long().to(self.device)
                        
                        if prompt_token.dim() == 3 and prompt_token.size(1) == 1:
                            prompt_token = prompt_token.squeeze(1)
                        elif prompt_token.dim() != 2:
                            raise ValueError(f"prompt_token 的维度应为 2D, 但得到 {prompt_token.dim()}D")
                        this_tts_speech = self.token2wav(
                            token=this_tts_speech_token,
                            prompt_token=prompt_token,
                            prompt_feat=prompt_feat,
                            embedding=flow_embedding,
                            uuid=uuid,
                            token_offset=token_offset,
                            finalize=False
                        )
                        token_offset += self.token_hop_len
                        self.token_offsets[uuid] = token_offset
                        yield {'tts_speech': this_tts_speech.cpu()}
                    if len(speech_tokens[token_offset:]) < required_tokens:
                        break
                this_tts_speech_token = torch.tensor(speech_tokens).unsqueeze(dim=0).long().to(self.device)
                this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=prompt_token,
                        prompt_feat=prompt_feat,
                        embedding=flow_embedding,
                        uuid=uuid,
                        token_offset=token_offset,
                        finalize=True,
                        speed=speed
                )
                self.token_offsets[uuid] = len(speech_tokens)
                yield {'tts_speech': this_tts_speech.cpu()}
        else:
            all_tokens = self.tts_speech_token_dict[uuid]
            if not all_tokens:
                import pdb; pdb.set_trace()
                raise ValueError("没有可生成的tokens")
            this_tts_speech_token = torch.tensor(all_tokens).long().unsqueeze(dim=0).to(self.device)
            
            if prompt_token.dim() == 3 and prompt_token.size(1) == 1:
                prompt_token = prompt_token.squeeze(1)
            elif prompt_token.dim() != 2:
                raise ValueError(f"prompt_token 的维度应为 2D, 但得到 {prompt_token.dim()}D")
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=flow_embedding,
                uuid=uuid,
                token_offset=0,
                finalize=True,
                speed=speed
            )
            self.token_offsets[uuid] = len(all_tokens)
            yield {'tts_speech': this_tts_speech.cpu()}
            
    def generate_speech_tokens(self, text, llm_embedding=torch.zeros(0, 192),
                               prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                               llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                               stream=False):
        """
        文本-speech token函数。

        参数:
            text (Tensor): 输入文本。
            llm_embedding (Tensor, optional): LLM 的嵌入向量。默认值为全零张量。
            prompt_text (Tensor, optional): 提示文本。默认值为空张量。
            llm_prompt_speech_token (Tensor, optional): LLM 提示speech token。默认值为空张量。
            stream (bool, optional): 是否以流式方式输出speech token。默认值为 False。

        返回:
            如果 `stream` 为 True，返回一个生成器，逐步输出speech token。
            如果 `stream` 为 False，返回所有生成的speech token。
        """
        # 生成唯一的会话ID，用于跟踪相关变量
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None

        # 启动生成speech token的线程
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()

        if stream:
            token_offset = 0
            while True:
                time.sleep(0.1)  # 等待新token生成
                # 检查是否有足够的token可以输出
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len:
                    # 获取新的token片段
                    new_tokens = self.tts_speech_token_dict[this_uuid][token_offset: token_offset + self.token_hop_len + self.flow.pre_lookahead_len]
                    token_offset += self.token_hop_len
                    this_speech_token = torch.tensor(new_tokens).unsqueeze(dim=0)
                    yield {'speech_token': this_speech_token.cpu()}
                # 检查LLM是否完成生成
                if self.llm_end_dict[this_uuid] and (len(self.tts_speech_token_dict[this_uuid]) - token_offset < self.token_hop_len + self.flow.pre_lookahead_len):
                    break
            p.join()
            # 处理剩余的token
            remaining_tokens = self.tts_speech_token_dict[this_uuid][token_offset:]
            if remaining_tokens:
                this_speech_token = torch.tensor(remaining_tokens).unsqueeze(dim=0)
                yield {'speech_token': this_speech_token.cpu()}
        else:
            # 非流式模式，等待LLM完成
            p.join()
            all_tokens = self.tts_speech_token_dict[this_uuid]
            this_speech_token = torch.tensor(all_tokens).unsqueeze(dim=0)
            yield {'speech_token': this_speech_token.cpu()}

        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)

    def text_to_speech_token(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        
        # import pdb
        # pdb.set_trace()
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        if stream is True:
            raise NotImplementedError
        else:
            # deal with all tokens
            p.join()
            all_tokens = self.tts_speech_token_dict[this_uuid]
            yield {'speech_token': all_tokens}

        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            
    def reset_state(self, uuid):
        with self.lock:
            self.tts_speech_token_dict.pop(uuid, None)
            self.hift_cache_dict.pop(uuid, None)
            self.token_offsets.pop(uuid, None)