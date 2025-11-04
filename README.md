# EMPATHY OMNI: ENABLING EMPATHETIC SPEECH RESPONSE GENERATION THROUGH LARGE LANGUAGE MODELS



[![arXiv](https://img.shields.io/badge/arXiv-2409.06666-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2508.18655)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/ictnlp/LLaMA-Omni)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Model-blue.svg)](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni)


We present EMPATHY OMNI, a speech–language model based on Qwen2.5-Instruct that supports low-latency, high-quality empathetic interaction, jointly generating textual and acoustic responses from speech inputs.

<div align="center"><img src="images/model.png" width="75%"/></div>





## Install

1. Clone this repository.

```shell
git clone https://github.com/W311411/Empathy-Omni
cd LLaMA-Omni
```

2. Install packages.

```shell
conda create -n Eomni python=3.10
conda activate Eomni
pip install pip==24.0
pip install -e .
```

3. Install `fairseq`.

```shell
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e . --no-build-isolation
```

4. Install `flash-attention`.

```shell
pip install flash-attn --no-build-isolation
```

## Quick Start

1. Download the `Llama-3.1-8B-Omni` model from 🤗[Huggingface](). 

2. Download the `Whisper-large-v3` model.

```shell
import whisper
model = whisper.load_model("large-v3", download_root="models/speech_encoder/")
```


## Local Inference


```shell
bash omni_speech/infer/run.sh omni_speech/infer/examples
```

