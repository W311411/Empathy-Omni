import sys
sys.path.insert(0, '/apdcephfs_nj7/share_303172353/haoyyywang/en-omni')
from omni_speech.train.train_tdm import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
