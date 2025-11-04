import os
import copy
import pdb
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import tokenizers
from torch.utils.data import Dataset
from PIL import Image
import whisper


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'speech' in sources[0]:
            # conv = conv_templates[self.conv_mode].copy()
            # conv.append_message(conv.roles[0], qs)
            # conv.append_message(conv.roles[1], None)
            # prompt = conv.get_prompt()
            # speech = whisper.load_audio(speech_file)
            # if self.input_type == "raw":
            #     speech = torch.from_numpy(speech)
            #     if self.model_config.speech_normalize:
            #         speech = torch.nn.functional.layer_norm(speech, speech.shape)
            # elif self.input_type == "mel":
            #     speech = whisper.pad_or_trim(speech)
            #     speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
            # input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
            # return input_ids, speech, torch.LongTensor([speech.shape[0]])

            qs = self.list_data_dict[i]["conversations"][0]["value"]
            speech_file = self.list_data_dict[i]['speech']
            speech_folder = self.data_args.speech_folder
            full_speech_path = os.path.join(speech_folder, speech_file)
            # speech = whisper.load_audio(os.path.join(speech_folder,speech_file))
            try:
                speech = whisper.load_audio(os.path.join(speech_folder,speech_file))
            except Exception as e:
                print(f"Error loading {speech_file}: {e}")
                print(os.path.join(speech_folder, speech_file))
                exit(0)
            
            if self.data_args.input_type == "raw":
                speech_length = torch.LongTensor([speech.shape[0]])
                # speech = whisper.pad_or_trim(speech)
                speech = torch.from_numpy(speech)
                # if self.model_config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
            elif self.data_args.input_type == "mel":
                # pdb.set_trace()
                raw_len = speech.shape[0]
                speech = whisper.pad_or_trim(speech)
                pad_len = speech.shape[0]
                speech = whisper.log_mel_spectrogram(speech, n_mels=self.data_args.mel_size).permute(1, 0)
                # pdb.set_trace()
                speech_length = round(raw_len / pad_len * 3000 + 0.5)
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        # pdb.set_trace()
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_speech=('speech' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'speech' in self.list_data_dict[i]:
            data_dict['speech'] = speech
            data_dict['speech_length'] = speech_length

        if 'units' in self.list_data_dict[i]:
            data_dict['tgt_unit'] = self.list_data_dict[i]['units']

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        # pdb.set_trace()
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'speech' in instances[0]:
            speeches = [instance['speech'] for instance in instances]
            speech_lengths = [instance['speech_length'] for instance in instances]
            # pdb.set_trace()
            # speeches = torch.nn.utils.rnn.pad_sequence(
            #     speeches,
            #     batch_first=True,
            #     padding_value=0)
            if all(x is not None and x.shape == speeches[0].shape for x in speeches):
                batch['speech'] = torch.stack(speeches).to(torch.bfloat16)
            else:
                batch['speech'] = speeches

            batch['speech_lengths'] = torch.tensor(speech_lengths)#.squeeze()
        
        if 'tgt_unit' in instances[0]:
            tgt_units = [torch.tensor(instance['tgt_unit']) for instance in instances]
            tgt_units = torch.nn.utils.rnn.pad_sequence(
                tgt_units,
                batch_first=True,
                padding_value=IGNORE_INDEX)
            batch['tgt_units'] = tgt_units

        return batch

