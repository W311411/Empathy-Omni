import argparse
import torch
import os
import sys
sys.path.insert(0, '/apdcephfs_nj7/share_303172353/haoyyywang/en-omni')
import json
from tqdm import tqdm
import whisper
from omni_speech.model.builder import load_pretrained_model
from omni_speech.utils import disable_torch_init
from omni_speech.datasets.preprocess import tokenizer_speech_token, preprocess_llama_3_v2
from torch.utils.data import Dataset, DataLoader
import pdb
import math
import soundfile as sf
import numpy as np

current_pythonpath = os.getenv('PYTHONPATH', '')
new_path = '/root/SpeechLLMs/'
if new_path not in current_pythonpath.split(os.pathsep):
    updated_pythonpath = current_pythonpath + os.pathsep + new_path if current_pythonpath else new_path
    os.environ['PYTHONPATH'] = updated_pythonpath


def merge_lora(args):
    tokenizer, model, _ = load_pretrained_model(args.model_path, args.model_base, is_lora=True, s2s=args.s2s)

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/speechllm_checkpoints/test2/lora_checkpoints")
    parser.add_argument("--model-base", type=str, default="/remote-home/yushengliao/yhwang/HuggingFace-Download-Accelerator/hf_hub/Llama-3.2-1B-Instruct")
    parser.add_argument("--save-model-path", type=str, default="/root/speechllm_checkpoints/test2/tmp")
    parser.add_argument("--s2s", action="store_true", default=False)

    args = parser.parse_args()

    merge_lora(args)
