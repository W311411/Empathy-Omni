#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import argparse
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass


# ================== 说话人嵌入（SpeechBrain ECAPA） ==================

def get_speaker_encoder(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    说话人相似度：使用 SpeechBrain ECAPA-TDNN（VoxCeleb 预训练）
    """
    from speechbrain.pretrained import EncoderClassifier
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    return spk_model


# ================== 音频预处理 & 嵌入提取 ==================

def load_audio(path, target_sr=16000):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # 单声道
    wav = torch.from_numpy(wav)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = torch.clamp(wav, -1.0, 1.0)
    return wav, target_sr


@torch.no_grad()
def emotion_embed(emotion_encoder, wav: torch.Tensor, device):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.to(device)

    out = emotion_encoder.extract_features(wav, None)["x"]
    if out.dim() == 3:
        feats = out.mean(dim=-1)
    elif out.dim() == 2:
        feats = out
    else:
        raise ValueError(f"Unexpected emotion features shape: {tuple(out.shape)}")

    vec = feats[0]
    vec = vec / (vec.norm(p=2) + 1e-9)
    return vec


@torch.no_grad()
def speaker_embed(spk_model, wav: torch.Tensor, device):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.to(device)
    emb = spk_model.encode_batch(wav).squeeze()
    emb = emb / (torch.norm(emb, p=2) + 1e-9)
    return emb


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sum(a * b).item())


# ================== fairseq 用户模块包装 ==================

@dataclass
class UserDirModule:
    user_dir: str


def get_emotion_encoder(encoder_fairseq_dir: str, emotion_encoder_ckpt: str, device: str):
    """
    加载 emotion2vec 模型，返回已设置到 device 的模型
    """
    import fairseq
    model_path = UserDirModule(encoder_fairseq_dir)
    fairseq.utils.import_user_module(model_path)
    model_ens, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [emotion_encoder_ckpt]
    )
    model = model_ens[0].to(device).eval()
    return model


# ================== 主流程 ==================

def main():
    parser = argparse.ArgumentParser(description="Compute emotion & speaker similarity.")
    parser.add_argument("--ref", type=str, required=True, help="参考音频路径")
    parser.add_argument("--gen_dir", type=str, required=True, help="生成音频目录")
    parser.add_argument("--pattern", type=str, default="*.wav", help="匹配生成音频的通配符，如 *.wav")
    parser.add_argument("--sr", type=int, default=16000, help="统一重采样到的采样率（建议与训练一致，如 16k）")
    parser.add_argument("--out", type=str, default="results.csv", help="输出CSV路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min_dur", type=float, default=0.2, help="最短时长(秒)，过短将跳过")

    # emotion2vec 加载参数
    parser.add_argument("--encoder_fairseq_dir", type=str, required=True, help="fairseq 用户模块目录")
    parser.add_argument("--emotion_encoder_ckpt", type=str, required=True, help="emotion2vec 模型 checkpoint 路径")

    args = parser.parse_args()
    device = args.device
    os.makedirs(Path(args.out).parent, exist_ok=True)

    print("Loading emotion2vec encoder ...")
    emo_enc = get_emotion_encoder(args.encoder_fairseq_dir, args.emotion_encoder_ckpt, device)

    print("Loading speaker encoder (ECAPA) ...")
    spk_enc = get_speaker_encoder(device=device)

    print(f"Loading reference: {args.ref}")
    ref_wav, _ = load_audio(args.ref, target_sr=args.sr)
    if ref_wav.numel() < int(args.min_dur * args.sr):
        raise ValueError("参考音频过短，无法计算相似度。")

    ref_emo = emotion_embed(emo_enc, ref_wav, device)
    ref_spk = speaker_embed(spk_enc, ref_wav, device)

    gen_paths = sorted(glob.glob(str(Path(args.gen_dir) / args.pattern)))
    if len(gen_paths) == 0:
        raise FileNotFoundError(f"未在 {args.gen_dir} 下匹配到 {args.pattern}")

    rows = [("file", "emotion_cosine", "speaker_cosine", "dur_sec")]
    for p in tqdm(gen_paths, desc="Processing"):
        try:
            wav, _ = load_audio(p, target_sr=args.sr)
            dur = wav.numel() / float(args.sr)
            if wav.numel() < int(args.min_dur * args.sr):
                print(f"[skip] {p} 音频过短 ({dur:.2f}s)")
                continue

            emo_vec = emotion_embed(emo_enc, wav, device)
            spk_vec = speaker_embed(spk_enc, wav, device)

            emo_sim = cosine_sim(ref_emo, emo_vec)
            spk_sim = cosine_sim(ref_spk, spk_vec)

            rows.append((p, f"{emo_sim:.6f}", f"{spk_sim:.6f}", f"{dur:.3f}"))
        except Exception as e:
            print(f"[error] {p}: {e}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Done. Results -> {args.out}")
    if len(rows) > 1:
        arr = np.array([[float(r[1]), float(r[2])] for r in rows[1:]], dtype=float)
        files = [r[0] for r in rows[1:]]

        emo_rank = np.argsort(-arr[:, 0])[:5]
        spk_rank = np.argsort(-arr[:, 1])[:5]



if __name__ == "__main__":
    main()
