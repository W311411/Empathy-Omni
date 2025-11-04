import os
import re
import json
import argparse
from typing import Tuple, Callable

import torch
import numpy as np
import librosa
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
import onnxruntime

from omni_speech.model.builder import load_pretrained_model
from omni_speech.datasets.preprocess import preprocess_llama_3_v1, preprocess_qwen_2_5_v1

from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.model import CosyVoice2Model
from cosyvoice.utils.frontend_utils import (
    contains_chinese, replace_blank, replace_corner_mark, remove_bracket,
    spell_out_number, split_paragraph
)

# -------------------------
# 常量配置（按需修改）
# -------------------------
COSYVOICE_MODEL = ""   # 例如 /workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B-VocalNet
MODEL_CHECKPOINT = ""
PROMPT_SPEECH = "v"  # 16k wav

# -------------------------
# ttsfrd / 文本正则化
# -------------------------
try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False


class SpeechTokenizer:
    def __init__(
        self,
        speech_tokenizer_model: str,
        feat_extractor: Callable,
        get_tokenizer: Callable,
        campplus_model: str,
        allowed_special: str = 'all',
        device: str = None
    ):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = get_tokenizer()
        self.allowed_special = allowed_special

        # onnxruntime 设置
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]

        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model, sess_options=option, providers=providers
        )
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )

        self.feat_extractor = feat_extractor
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            root_dir = os.path.dirname(os.path.abspath(__file__))
            ok = self.frd.initialize(f'{root_dir}/pretrained_models/CosyVoice-ttsfrd/resource')
            assert ok, 'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False)
            self.en_tn_model = EnNormalizer()

    def extract_speech_token(self, speech: torch.Tensor, max_duration_sec: int = 30):
        sample_rate = 16000
        max_samples = max_duration_sec * sample_rate

        total_samples = speech.shape[1]
        num_chunks = (total_samples + max_samples - 1) // max_samples

        speech_tokens = []
        speech_token_lengths = []

        for i in range(num_chunks):
            start = i * max_samples
            end = min(start + max_samples, total_samples)
            chunk = speech[:, start:end]
            feat = whisper.log_mel_spectrogram(chunk, n_mels=128)  # [80, T]

            inputs = {
                self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                self.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
            }
            out = self.speech_tokenizer_session.run(None, inputs)[0]
            chunk_token = torch.tensor(out.flatten().tolist(), dtype=torch.int32).unsqueeze(0).to(self.device)
            chunk_len = torch.tensor([chunk_token.shape[1]], dtype=torch.int32).to(self.device)

            speech_tokens.append(chunk_token)
            speech_token_lengths.append(chunk_len)

        speech_token = torch.cat(speech_tokens, dim=1)
        speech_token_len = torch.cat(speech_token_lengths, dim=0).sum().unsqueeze(0)
        return speech_token, speech_token_len

    def save_tokens(self, speech_token: torch.Tensor, speech_token_len: torch.Tensor, save_path: str):
        np.save(save_path, {
            'speech_token': speech_token.cpu().numpy(),
            'speech_token_len': speech_token.cpu().numpy()
        })
        print(f"Tokens saved to {save_path}")

    def load_tokens(self, load_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(load_path, allow_pickle=True).item()
        st = torch.tensor(data['speech_token'])
        stl = torch.tensor(data['speech_token_len'])
        print(f"Tokens loaded from {load_path}")
        return st, stl

    def extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        emb = self.campplus_session.run(None, {
            self.campplus_session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()
        })[0].flatten().tolist()
        return torch.tensor([emb]).to(self.device)

    def extract_text_token(self, text):
        ids = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        token = torch.tensor([ids], dtype=torch.int32).to(self.device)
        token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device)
        return token, token_len

    def text_normalize(self, text, split=True):
        text = text.strip()
        # 如果包含额外special token，直接返回
        for token in self.tokenizer.special_tokens['additional_special_tokens']:
            if token in text:
                return text if not split else [text]

        if contains_chinese(text):
            if self.use_ttsfrd:
                texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
                text = ''.join(texts)
            else:
                text = self.zh_tn_model.normalize(text).replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。").replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(split_paragraph(
                    text, lambda t: self.tokenizer.encode(t, allowed_special=self.allowed_special),
                    "zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False
                ))
        else:
            if self.use_ttsfrd:
                texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
                text = ''.join(texts)
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(split_paragraph(
                    text, lambda t: self.tokenizer.encode(t, allowed_special=self.allowed_special),
                    "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False
                ))
        return text if not split else texts


class CosyvoiceVocoder:
    def __init__(self, frontend: SpeechTokenizer, model: CosyVoice2Model, sample_rate: int = 24000):
        self.frontend = frontend
        self.model = model
        self.sample_rate = sample_rate

    def inference_zero_shot(self, speech_token, prompt_token: torch.Tensor,
                            prompt_feat: torch.Tensor, embedding: torch.Tensor,
                            stream=False, speed=1.0):
        # 形状调整
        if speech_token.dim() == 3 and speech_token.size(1) == 1:
            speech_token = speech_token.squeeze(1)
        if speech_token.dim() != 2:
            raise ValueError(f"speech_token must be 2D, got {speech_token.dim()}D")

        if prompt_token.dim() == 3 and prompt_token.size(1) == 1:
            prompt_token = prompt_token.squeeze(1)
        if prompt_token.dim() != 2:
            raise ValueError(f"prompt_token must be 2D, got {prompt_token.dim()}D")

        if prompt_feat.dim() != 3:
            raise ValueError(f"prompt_feat must be 3D, got {prompt_feat.dim()}D")

        model_input = {
            'speech_tokens': speech_token,
            'flow_embedding': embedding,
            'prompt_token': prompt_token,
            'prompt_feat': prompt_feat,
            'stream': stream,
            'speed': speed
        }
        for out in self.model.tts_direct(**model_input):
            yield out


class Eomni:
    def __init__(self, model_name_or_path: str, vocoder_path: str = COSYVOICE_MODEL, s2s: bool = True, **kwargs):
        self.s2s = s2s
        self.model_name_or_path = model_name_or_path
        self.vocoder_path = vocoder_path
        self.empty = True

        self.temperature = kwargs.get('temperature', 0)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)
        self.top_p = kwargs.get('top_p', 0.1)
        self.streaming = kwargs.get('streaming', False)

        self.audio_dir = None
        self.idx = 0

    def __initilize__(self):
        if self.empty:
            self.empty = False
            self.tokenizer, self.model, _ = load_pretrained_model(self.model_name_or_path, s2s=self.s2s)
            self.__init_vocoder__()

    def __init_vocoder__(self):
        model_dir = self.vocoder_path
        with open(f'{model_dir}/cosyvoice.yaml', 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        frontend = SpeechTokenizer(
            speech_tokenizer_model=f"{model_dir}/speech_tokenizer_v2.onnx",
            feat_extractor=configs['feat_extractor'],
            get_tokenizer=configs["get_tokenizer"],
            allowed_special=configs['allowed_special'],
            campplus_model=f"{model_dir}/campplus.onnx"
        )

        model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'])
        model.load(
            f'{model_dir}/llm.pt',
            f'{model_dir}/flow.pt',
            f'{model_dir}/hift.pt'
        )
        self.cosy_vocoder = CosyvoiceVocoder(frontend=frontend, model=model)

        # 准备零样本提示
        prompt_sr = 16000
        prompt_speech_16k = self.postprocess(load_wav(PROMPT_SPEECH, prompt_sr))
        prompt_token, prompt_token_len = frontend.extract_speech_token(prompt_speech_16k)

        # 24k 特征
        resample_rate = 24000
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=prompt_sr, new_freq=resample_rate)(prompt_speech_16k)
        speech_feat, speech_feat_len = frontend.extract_speech_feat(prompt_speech_resample)

        if resample_rate == 24000:
            token_len = min(int(speech_feat.shape[1] / 2), prompt_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            prompt_token, prompt_token_len[:] = prompt_token[:, :token_len], token_len

        embedding = frontend.extract_spk_embedding(prompt_speech_16k)

        self.prompt_token = prompt_token
        self.speech_feat = speech_feat
        self.embedding = embedding

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440, max_val=0.8):
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
        )
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        speech = torch.concat([speech, torch.zeros(1, int(24000 * 0.2))], dim=1)
        return speech

    def set_audio_dir(self, audio_dir):
        self.audio_dir = audio_dir
        os.makedirs(audio_dir, exist_ok=True)

    def __call__(self, messages: list) -> dict:
        """messages: [{'from':'human','value':'<speech>','path': '.../xxx.wav'}]"""
        assert len(messages) > 0 and 'path' in messages[0], 
        audio_path = messages[0]['path']

        speech_np = whisper.load_audio(audio_path)
        wav = torch.from_numpy(speech_np)

        raw_len = len(speech_np)
        speech_pad = whisper.pad_or_trim(speech_np)
        mel = whisper.log_mel_spectrogram(speech_pad, n_mels=128).permute(1, 0).unsqueeze(0)
        speech_length = torch.LongTensor([round(raw_len / len(speech_pad) * 3000 + 0.5)])


        conversation = messages  
        if 'qwen' in self.model_name_or_path.lower():
            input_ids = preprocess_qwen_2_5_v1([conversation], self.tokenizer, True, 4096)['input_ids']
            input_ids = torch.cat([input_ids.squeeze(),
                                   torch.tensor([198, 151644, 77091, 198], device=input_ids.device)]).unsqueeze(0)
        else:
            input_ids = preprocess_llama_3_v1([conversation], self.tokenizer, True, 4096)['input_ids']
            input_ids = torch.cat([input_ids.squeeze(),
                                   torch.tensor([128006, 78191, 128007, 271], device=input_ids.device)]).unsqueeze(0)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_ids = input_ids.to(device=device, non_blocking=True)
        speech_tensor = mel.to(dtype=torch.float16 if device == 'cuda' else torch.float32, device=device, non_blocking=True)
        speech_length = speech_length.to(device=device, non_blocking=True)
        wav = wav.to(dtype=torch.float16 if device == 'cuda' else torch.float32, device=device, non_blocking=True)
        wav_length = torch.LongTensor([wav.shape[0] // 320]).to(device=device, non_blocking=True)

        with torch.inference_mode():
            if self.s2s:
                outputs = self.model.generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    do_sample=(self.temperature > 0),
                    temperature=self.temperature,
                    top_p=(self.top_p if self.top_p is not None else 0.0),
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                    streaming_unit_gen=False,
                    wav=wav,
                    wav_lengths=wav_length,
                )
                output_ids, output_units = outputs
            else:
                outputs = self.model.generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    do_sample=(self.temperature > 0),
                    temperature=self.temperature,
                    top_p=(self.top_p if self.top_p is not None else 0.0),
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                    wav=wav,
                    wav_lengths=wav_length,
                )
                output_ids = outputs[0]

        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        result = {"text": output_text}

        if not self.s2s:
            return result

        # S2S: 语音解码
        output_units = output_units[:, 1:-1]  # 去掉<bos>/<eos>
        audio_file = None
        if not self.streaming:
            for out in self.cosy_vocoder.inference_zero_shot(
                speech_token=output_units,
                prompt_token=self.prompt_token,
                prompt_feat=self.speech_feat,
                embedding=self.embedding,
                stream=self.streaming,
                speed=1.0
            ):
                speech = out['tts_speech']
                base_name = os.path.basename(audio_path)
                audio_file = os.path.join(self.audio_dir, base_name)
                torchaudio.save(audio_file, speech.cpu(), self.cosy_vocoder.sample_rate)

        if audio_file:
            result['audio'] = audio_file
        return result


def load_audio_paths(txt_file: str, prefix: str) -> list:
    paths = []
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                p = line.strip()
                if p:
                    paths.append(os.path.join(prefix, p))
        print(f"已从 {txt_file} 读取 {len(paths)} 条路径")
        return paths
    except FileNotFoundError:
        print(f"错误: 找不到文件 {txt_file}")
        return []
    except Exception as e:
        print(f"发生错误: {e}")
        return []


if __name__ == "__main__":
    import argparse, json, os
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='single-audio inference')
    parser.add_argument('--query_audio', type=str, required=True,
                        help='输入音频路径（wav/其它 whisper 可读格式）')
    parser.add_argument('--s2s', action='store_true', default=True,
                        help='是否进行语音到语音生成（默认开启）')
    parser.add_argument('--save_dir', type=str, default=f"{MODEL_CHECKPOINT}/male",
                        help='输出音频保存目录（对 s2s 生效）')
    args = parser.parse_args()

    print("Initialized model")
    engine = Eomni(MODEL_CHECKPOINT, s2s=args.s2s)
    engine.__initilize__()                 # 保持原方法名
    engine.set_audio_dir(args.save_dir)

    # 单条音频推理
    audio_path = args.query_audio
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"找不到输入音频: {audio_path}")

    messages = [{"from": "human", "value": "<speech>", "path": audio_path}]
    resp = engine(messages)

    # 输出结果（不写入文件）
    print("\n=== Inference Result ===")
    print("Input audio:", audio_path)
    print("Text:", resp.get("text", ""))

    # s2s 时会有合成语音文件路径
    if "audio" in resp:
        print("TTS audio saved to:", resp["audio"])
    else:
        print("No TTS audio generated")
