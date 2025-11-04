# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import copy
import torch
import transformers
import tokenizers
import pdb
import os
import glob
import pandas as pd

from typing import Dict, Sequence

from omni_speech.constants import IGNORE_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
from omni_speech import conversation as conversation_lib
from omni_speech.model import *
from omni_speech.arguments import DataArguments

from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

# 
def tokenizer_speech_token(prompt, tokenizer, speech_token_index=SPEECH_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<speech>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [speech_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for sentence in sources:
        for key, value in sentence.items():
            if DEFAULT_SPEECH_TOKEN in value:
                value = value.replace(DEFAULT_SPEECH_TOKEN, '').strip()
                value = DEFAULT_SPEECH_TOKEN + '\n' + value
                value = value.strip()

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_speech:
                round_len = len(tokenizer_speech_token(rou, tokenizer))
                instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    # pdb.set_trace()
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_speech:
                round_len = len(tokenizer_speech_token(rou, tokenizer))
                instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # FIXME: tokenizer bug
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_SPEECH_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_SPEECH_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_speech_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess_qwen_2_5(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]  # Skip the first one if it is not from human

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
            # message = f"<|im_start|>{role}\n{sentence['value']}<|im_end|>"
            # conv.append_message(role, message)
        
        conversations.append(conv.get_prompt())

    # pdb.set_trace()
    # Tokenize conversations
    if has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    # Mask targets according to Qwen 2.5 separator style
    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN_2_5
    sep = "<|im_start|>assistant\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        parts = conversation.split(sep)
        parts[0] += sep  
        # pdb.set_trace()

        if has_speech:
            conversation_len = len(tokenizer_speech_token(conversation, tokenizer))
            instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 1
        else:
            conversation_len = len(tokenizer(conversation).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
        cur_len += conversation_len
        target[cur_len:] = IGNORE_INDEX
        
        # if cur_len < tokenizer.model_max_length:
        #     if cur_len != total_len:
        #         target[:] = IGNORE_INDEX
        #         print(
        #             f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
        #             f" (ignored)"
        #         )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_qwen_2_5_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False,
    max_len=2048,
    system_message: str = "You are Qwen, created by Alibaba Cloud. You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language and speech. Please provide a concise response in one or two sentences."
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    if has_speech:
        tokenizer.add_tokens(["<speech>"], special_tokens=True)
    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    unmask_tokens = []
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    input_ids, targets = [], []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        all_convs = [{"role": "system", "content": system_message}]
        
        for conv in source[:-1]:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            conv = [{"role": role, "content": content}]
            all_convs += conv

        input_id += tokenizer.apply_chat_template(all_convs, tokenize=True)
        # tokenizer.apply_chat_template(all_convs, tokenize=False, add_generation_prompt=True)
        target += [IGNORE_INDEX] * len(input_id)

        conv = source[-1]
        try:
            role = conv["role"]
            content = conv["content"]
        except:
            role = conv["from"]
            content = conv["value"]

        role = roles.get(role, role)
        conv = [{"role": role, "content": content}]
        
        encode_id = tokenizer.apply_chat_template(conv, tokenize=True)[21:-1] # conv无system，故qwen会额外添加system prompt，重复遂去除
        encode_id.append(eot_id)

        input_id += encode_id
        target += [IGNORE_INDEX] * 3
        target += encode_id[3:]
        
        # pdb.set_trace()
        
        # print(tokenizer.decode(input_id, skip_special_tokens=False))

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == speech_token_index:
                input_id[idx] = SPEECH_TOKEN_INDEX
        
        input_ids.append(input_id)
        targets.append(target)
    # pdb.set_trace()
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def preprocess_qwen_2_5_v2(
    conversation_text: str,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False,
    has_instruct: bool = False,
    max_len=2048,
    system_message: str = "You are Qwen, created by Alibaba Cloud. You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language and speech."
) -> Dict:
    tts = False
    if tts is True:
     
        system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language and speech.Please repeat exactly what I input next, without adding any extra explanations or modifications."
    
    if isinstance(conversation_text, dict) and "conversation" in conversation_text:
        conversations = conversation_text["conversation"]  # 提取对话列表
    elif isinstance(conversation_text, list):
        conversations = conversation_text  # 如果直接传入的是对话列表
    else:
        raise ValueError(f"不支持的数据格式: {type(conversation_text)}")
    tokenizer = copy.deepcopy(tokenizer)
    if has_speech:
        tokenizer.add_tokens(["<speech>"], special_tokens=True)
    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    unmask_tokens = []
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    input_ids, targets = [], []
    conversations = conversation_text

    full_conversation_history = [{"role": "system", "content": system_message}]
    
    input_id, target = [], []
    for conv in conversations[:-1]:#处理除了最后一轮
        try:
            role = conv["role"].lower()
            content = conv["content"]
        except:
            role = conv["from"]
            content = conv["value"]

        conv = [{"role": role, "content": content}]
        full_conversation_history += conv
    # pdb.set_trace()

    input_id += tokenizer.apply_chat_template(full_conversation_history, tokenize=True)
    # tokenizer.apply_chat_template(all_convs, tokenize=False, add_generation_prompt=True)
    target += [IGNORE_INDEX] * len(input_id)
    conv = conversations[-1]
    try:
        role = conv["role"].lower()
        content = conv["content"]
    except:
        role = conv["from"]
        content = conv["value"]

    conv = [{"role": role, "content": content}]
    
    encode_id = tokenizer.apply_chat_template(conv, tokenize=True)[21:-1] # conv无system，故qwen会额外添加system prompt，重复遂去除
    encode_id.append(eot_id)

    input_id += encode_id
    target += [IGNORE_INDEX] * 3
    target += encode_id[3:]
    
    abc = tokenizer.decode(input_id, skip_special_tokens=False)
    # print(abc)
    assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
    for idx, encode_id in enumerate(input_id):
        if encode_id in unmask_tokens_idx:
            target[idx] = encode_id
        if encode_id == speech_token_index:
            # print(idx)
            input_id[idx] = SPEECH_TOKEN_INDEX
    
    input_ids.append(input_id)
    targets.append(target)
    # pdb.set_trace()
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def preprocess_qwen_3_v1(
    conversation_text: str,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False,
    has_instruct: bool = False,
    max_len=2048,
    system_message: str = "You are Qwen, an advanced language and empathetic speech assistant. created by Alibaba Cloud. Listen carefully to the user's speech to understand both content and emotional state (tone, pace, mood). Acknowledge their feelings first, then provide supportive responses. Generate speech replies that match the appropriate emotional tone."
) -> Dict:
    tts = False
    if tts is True:
     
        system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language and speech.Please repeat exactly what I input next, without adding any extra explanations or modifications."
    
    if isinstance(conversation_text, dict) and "conversation" in conversation_text:
        conversations = conversation_text["conversation"]  # 提取对话列表
    elif isinstance(conversation_text, list):
        conversations = conversation_text  # 如果直接传入的是对话列表
    else:
        raise ValueError(f"不支持的数据格式: {type(conversation_text)}")
    tokenizer = copy.deepcopy(tokenizer)
    if has_speech:
        tokenizer.add_tokens(["<speech>"], special_tokens=True)
    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    unmask_tokens = []
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    input_ids, targets = [], []
    conversations = conversation_text

    full_conversation_history = [{"role": "system", "content": system_message}]
    
    input_id, target = [], []
    for conv in conversations[:-1]:#处理除了最后一轮
        try:
            role = conv["role"].lower()
            content = conv["content"]
        except:
            role = conv["from"]
            content = conv["value"]

        conv = [{"role": role, "content": content}]
        full_conversation_history += conv
    # pdb.set_trace()

    input_id += tokenizer.apply_chat_template(full_conversation_history, tokenize=True, enable_thinking=False)
    # tokenizer.apply_chat_template(all_convs, tokenize=False, add_generation_prompt=True)
    target += [IGNORE_INDEX] * len(input_id)
    conv = conversations[-1]
    try:
        role = conv["role"].lower()
        content = conv["content"]
    except:
        role = conv["from"]
        content = conv["value"]
#tokenizer.decode(encode_id, skip_special_tokens=False)
    conv = [{"role": role, "content": content}]
    encode_id = tokenizer.apply_chat_template(conv, tokenize=True, enable_thinking=False)
    encode_id.append(eot_id)

    input_id += encode_id
    target += [IGNORE_INDEX] * 3
    target += encode_id[3:]
    
    # abc = tokenizer.decode(target, skip_special_tokens=False)
    # print(abc)
    assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
    for idx, encode_id in enumerate(input_id):
        if encode_id in unmask_tokens_idx:
            target[idx] = encode_id
        if encode_id == speech_token_index:
            # print(idx)
            input_id[idx] = SPEECH_TOKEN_INDEX
    
    input_ids.append(input_id)
    targets.append(target)
    # pdb.set_trace()
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )



def preprocess_qwen_2_5_v3(
    conversation_text: str,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False,
    has_instruct: bool = True,
    max_len=2048,
    system_message: str = "You are Qwen, created by Alibaba Cloud. You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language and speech."
) -> Dict:
    tts = False
    if tts is True:
        system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language and speech.Please repeat exactly what I input next, without adding any extra explanations or modifications."
    
    tokenizer = copy.deepcopy(tokenizer)
    if has_speech:
        tokenizer.add_tokens(["<speech>"], special_tokens=True)

    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    unmask_tokens = []
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    input_ids, targets = [], []
    conversations = conversation_text

    full_conversation_history = [{"role": "system", "content": system_message}]
    # import pdb; pdb.set_trace()
    input_id, target = [], []
    conv = conversations[-1]
    try:
        role = conv["role"].lower()
        content = conv["content"]
    except:
        role = conv["from"]
        content = conv["value"]

    conv = [{"role": "user", "content": content}]
    full_conversation_history += conv   
    
    input_id += tokenizer.apply_chat_template(full_conversation_history, tokenize=True)
    # tokenizer.apply_chat_template(all_convs, tokenize=False, add_generation_prompt=True)
    target += [IGNORE_INDEX] * len(input_id)
    # import pdb;pdb.set_trace()  
    conv = conversations[-1]
    try:
        role = conv["role"].lower()
        content = conv["content"]
    except:
        role = conv["from"]
        content = conv["value"]

    conv = [{"role": "assistant", "content": content}]
    
    encode_id = tokenizer.apply_chat_template(conv, tokenize=True)[21:-1] # conv无system，故qwen会额外添加system prompt，重复遂去除
    encode_id.append(eot_id)

    input_id += encode_id
    target += [IGNORE_INDEX] * 3
    target += encode_id[3:]
    abc = tokenizer.decode(input_id, skip_special_tokens=False)
    # print(abc)
    assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
    for idx, encode_id in enumerate(input_id):
        if encode_id in unmask_tokens_idx:
            target[idx] = encode_id
        if encode_id == speech_token_index:
            # print(idx)
            input_id[idx] = SPEECH_TOKEN_INDEX
    
    input_ids.append(input_id)
    targets.append(target)
    # pdb.set_trace()
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    # pdb.set_trace()
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        assert len(source) == 2, "now only support single-turn conversation"

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = "<|start_header_id|>" + conv.roles[1] + "<|end_header_id|>\n\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        parts = conversation.split(sep)
        parts[0] += sep
        # pdb.set_trace()

        if has_speech:
            conversation_len = len(tokenizer_speech_token(conversation, tokenizer))
            instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 1
        else:
            conversation_len = len(tokenizer(conversation).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
        cur_len += conversation_len
        target[cur_len:] = IGNORE_INDEX
    
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = True,
    max_len=2048,
    system_message: str = "You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language and speech.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    if has_speech:
        tokenizer.add_tokens(["<speech>"], special_tokens=True)
    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = []
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        all_convs = [{"role": "system", "content": system_message}]
        for conv in source[:-1]:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            conv = [{"role": role, "content": content}]
            all_convs += conv

        input_id += tokenizer.apply_chat_template(all_convs, tokenize=True)
        target += [IGNORE_INDEX] * len(input_id)
        if input_id[-4:] == [128006, 78191, 128007, 271]:
            input_id = input_id[:-4]
            target = target[:-4]

        conv = source[-1]
        try:
            role = conv["role"]
            content = conv["content"]
        except:
            role = conv["from"]
            content = conv["value"]

        role =  roles.get(role, role)
        conv = [{"role" : role, "content" : content}]
        # First is bos token we don't need here
        encode_id = tokenizer.apply_chat_template(conv, tokenize=True)[1:]
        # pdb.set_trace()
        if encode_id[:4] == [128006, 9125, 128007, 271]:
            if eot_id in encode_id:
                eot_index = encode_id.index(eot_id)
                encode_id = encode_id[eot_index + 1:]
        if encode_id[-4:] == [128006, 78191, 128007, 271]:
            encode_id = encode_id[:-4]

        input_id += encode_id
        target += [IGNORE_INDEX] * 4
        target += encode_id[4:]

        # Convert input_ids back to text
        # text = tokenizer.decode(input_id, skip_special_tokens=True)
        
       
        # Mask the human words
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == speech_token_index:
                input_id[idx] = SPEECH_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def preprocess_llama_3_v2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = True,
    max_len=2048,
    system_message: str = "You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language and speech.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    if has_speech:
        tokenizer.add_tokens(["<speech>"], special_tokens=True)
    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = []
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        all_convs = [{"role": "system", "content": system_message}]
        for conv in source[:-1]:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            conv = [{"role": role, "content": content}]
            all_convs += conv

        input_id += tokenizer.apply_chat_template(all_convs, tokenize=True)
        target += [IGNORE_INDEX] * len(input_id)

        conv = source[-1]
        try:
            role = conv["role"]
            content = conv["content"]
        except:
            role = conv["from"]
            content = conv["value"]

        role =  roles.get(role, role)
        conv = [{"role" : role, "content" : content}]
        # First is bos token we don't need here
        encode_id = tokenizer.apply_chat_template(conv, tokenize=True)[1:]
        # pdb.set_trace()
        if eot_id in encode_id:
            eot_index = encode_id.index(eot_id)
            encode_id = encode_id[eot_index + 1:]
        input_id += encode_id
        target += [IGNORE_INDEX] * 4
        target += encode_id[4:]

        # Convert input_ids back to text
        # text = tokenizer.decode(encode_id, skip_special_tokens=False)
        # pdb.set_trace()
       
        # Mask the human words
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == speech_token_index:
                input_id[idx] = SPEECH_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    # pdb.set_trace()
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    import pdb; pdb.set_trace()
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False,
    has_instruct: bool = False,
    model_version: str = "llama_3_1",
    tts:bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    return preprocess_qwen_2_5_v2(sources, tokenizer, has_speech=has_speech)

