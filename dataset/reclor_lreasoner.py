import json
import os
import random
from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from general_util.logger import get_child_logger

logger = get_child_logger("ReClor.LReasoner")

_tokenizer: PreTrainedTokenizer


def initializer(tokenizer: PreTrainedTokenizer):
    global _tokenizer
    _tokenizer = tokenizer


def read_examples(file_path: str, ques_types, extend_contexts, negative_contexts, negative_extend_contexts):
    examples = []

    data = json.load(open(file_path, "r"))

    assert len(data) == len(ques_types) == len(extend_contexts) == len(negative_extend_contexts)

    _a = 0
    _b = 0
    for item_id, item in enumerate(data):
        context = item["context"]
        answers = item["answers"]
        question = item["question"]
        label = item["label"] if "label" in item else -1

        id_string = item["id_string"]

        _r = random.random()
        if label > 2:
            _a += 1
            contras_contexts = [negative_contexts[item_id][0], context]
            contras_label = 1
            contras_extend_context = [negative_extend_contexts[item_id][0], extend_contexts[item_id][label]]
        else:
            _b += 1
            contras_contexts = [context, negative_contexts[item_id][0]]
            contras_label = 0
            contras_extend_context = [extend_contexts[item_id][label], negative_extend_contexts[item_id][0]]

        assert len(extend_contexts[item_id]) == len(answers), extend_contexts[item_id]

        examples.append({
            "example_id": id_string,
            "question": question,
            "context": [context] * len(answers),
            "answers": answers,
            "label": label,
            "ques_type": ques_types[item_id],
            "extend_context": extend_contexts[item_id],
            "contras_context": contras_contexts,
            "contras_label": contras_label,
            "contras_answer": [answers[label], answers[label]],
            "contras_extend_context": contras_extend_context
        })

    logger.info(f"Contrastive learning labels ratio: {_a} / {_b} = {_a * 1.0 / _b}")
    return examples


def _convert_example_into_feature(example, max_seq_length: int, whether_extend_context: bool):
    context = example["context"]
    question = example["question"]
    answers = example["answers"]
    extend_contexts = example["extend_context"]

    input_id_ls = []
    attention_mask_ls = []
    token_type_id_ls = []
    for op_id, (op, extend_context) in enumerate(zip(answers, extend_contexts)):
        _text_b = question + _tokenizer.sep_token + op
        if whether_extend_context:
            _text_b = _text_b + _tokenizer.sep_token + extend_context

        tokenizer_outputs = _tokenizer(context[op_id], _text_b, max_length=max_seq_length, padding=PaddingStrategy.MAX_LENGTH,
                                       truncation=TruncationStrategy.LONGEST_FIRST)
        input_id_ls.append(tokenizer_outputs["input_ids"])
        attention_mask_ls.append(tokenizer_outputs["attention_mask"])
        if "token_type_ids" in tokenizer_outputs:
            token_type_id_ls.append(tokenizer_outputs["token_type_ids"])

    con_input_id_ls = []
    con_attention_mask_ls = []
    con_token_type_id_ls = []
    for c_id, (c_context, c_answer, c_extend_context) in enumerate(zip(example["contras_context"], example["contras_answer"],
                                                                       example["contras_extend_context"])):
        _text_b = question + _tokenizer.sep_token + c_answer
        if whether_extend_context:
            _text_b = _text_b + c_extend_context

        tokenizer_outputs = _tokenizer(c_context, _text_b, max_length=max_seq_length, padding=PaddingStrategy.MAX_LENGTH,
                                       truncation=TruncationStrategy.LONGEST_FIRST)

        con_input_id_ls.append(tokenizer_outputs["input_ids"])
        con_attention_mask_ls.append(tokenizer_outputs["attention_mask"])
        if "token_type_ids" in tokenizer_outputs:
            con_token_type_id_ls.append(tokenizer_outputs["token_type_ids"])

    outputs = {
        "input_ids": input_id_ls,
        "attention_mask": attention_mask_ls,
        "label": example["label"],
        "con_input_ids": con_input_id_ls,
        "con_attention_mask": con_attention_mask_ls,
        "con_label": example["contras_label"]
    }
    if token_type_id_ls:
        assert len(con_token_type_id_ls) > 0
        outputs["token_type_ids"] = token_type_id_ls
        outputs["con_token_type_ids"] = con_token_type_id_ls

    return outputs


def data2tensor(features):
    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    label = torch.tensor([f["label"] for f in features], dtype=torch.long)

    con_input_ids = torch.tensor([f["con_input_ids"] for f in features], dtype=torch.long)
    con_attention_mask = torch.tensor([f["con_attention_mask"] for f in features], dtype=torch.long)
    con_label = torch.tensor([f["con_label"] for f in features], dtype=torch.long)

    if "token_type_ids" in features[0]:
        token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
        con_token_type_ids = torch.tensor([f["con_token_type_ids"] for f in features], dtype=torch.long)
    else:
        token_type_ids = None
        con_token_type_ids = None

    logger.info(f"Input Size: {input_ids.size()}")
    logger.info(f"Contrastive learning Size: {con_input_ids.size()}")

    if token_type_ids is None:
        return input_ids, attention_mask, label, con_input_ids, con_attention_mask, con_label
    else:
        return input_ids, attention_mask, token_type_ids, label, con_input_ids, con_attention_mask, con_token_type_ids, con_label


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, whether_extend_context: bool,
                                   data_dir: str, version: int = 1, negative_version: int = 1, negative_extend_version: int = 91,
                                   num_workers: int = 16):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_lreason_{max_seq_length}_{whether_extend_context}_{version}_{negative_version}_{negative_extend_version}"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        examples, features, tensors = torch.load(cached_file_path)
        return examples, features, tensors

    _ques_type_tem = "{}_ques_types.npy"
    _extended_context_tem = "{}_extended_context_cp_v{}.npy"
    _negative_context_tem = "{}_negative_context_cp_v{}.npy"

    _file_type = ""
    if "train" in file_path:
        _file_type = "train"
    elif "val" in file_path:
        _file_type = "val"
    elif "test" in file_path:
        _file_type = "test"

    ques_types = np.load(os.path.join(data_dir, _ques_type_tem.format(_file_type)))
    extend_contexts = np.load(os.path.join(data_dir, _extended_context_tem.format(_file_type, str(version))), allow_pickle=True)
    negative_contexts = np.load(os.path.join(data_dir, _negative_context_tem.format(_file_type, str(negative_version))), allow_pickle=True)
    negative_extend_contexts = np.load(os.path.join(data_dir, _extended_context_tem.format(_file_type, str(negative_extend_version))),
                                       allow_pickle=True)

    examples = read_examples(file_path, ques_types, extend_contexts, negative_contexts, negative_extend_contexts)
    with Pool(num_workers, initializer=initializer, initargs=(tokenizer,)) as p:
        _annotate = partial(_convert_example_into_feature, max_seq_length=max_seq_length, whether_extend_context=whether_extend_context)
        features = list(tqdm(
            p.imap(_annotate, examples, chunksize=32),
            total=len(examples),
            desc='converting examples to features:'
        ))

    logger.info("Transform features into tensors...")
    tensors = data2tensor(features)

    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((examples, features, tensors), cached_file_path)

    return examples, features, tensors


class LReasonerCollator:
    def __init__(self, add_contras: bool = True, add_orig: bool = True):
        assert add_contras or add_orig, "There should be at least one option is ``True`` in ``add_contras`` and ``add_orig``."
        self.add_contras = add_contras
        self.add_orig = add_orig

    def __call__(self, batch: List[Tuple[Tensor, ...]]) -> Dict[str, Tensor]:
        if len(batch[0]) == 6:
            input_ids, attention_mask, label, con_input_ids, con_attention_mask, con_label = list(zip(*batch))
            token_type_ids = None
            con_token_type_ids = None
        elif len(batch[0]) == 8:
            (input_ids, attention_mask, token_type_ids, label,
             con_input_ids, con_attention_mask, con_token_type_ids, con_label) = list(zip(*batch))
        else:
            raise RuntimeError()

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(label, dim=0)

        con_input_ids = torch.stack(con_input_ids, dim=0)
        con_attention_mask = torch.stack(con_attention_mask, dim=0)
        con_labels = torch.stack(con_label, dim=0)

        outputs = {}
        if self.add_orig:
            outputs["input_ids"] = input_ids
            outputs["attention_mask"] = attention_mask
            outputs["labels"] = labels
            if token_type_ids is not None:
                outputs["token_type_ids"] = torch.stack(token_type_ids, dim=0)
        if self.add_contras:
            outputs["con_input_ids"] = con_input_ids
            outputs["con_attention_mask"] = con_attention_mask
            outputs["con_labels"] = con_labels
            if con_token_type_ids is not None:
                outputs["con_token_type_ids"] = torch.stack(con_token_type_ids, dim=0)

        return outputs
