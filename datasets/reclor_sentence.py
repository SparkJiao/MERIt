import json
import os
from nltk import sent_tokenize
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

import torch
from torch import Tensor
from typing import Tuple, Callable, Union, Dict, List
from torch.utils.data import TensorDataset, Dataset
from transformers import PreTrainedTokenizer, RobertaTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy, TensorType
from collections import Counter

from general_util.logger import get_child_logger

logger = get_child_logger("ReClor.Sentence")


def is_bpe(_tokenizer: PreTrainedTokenizer):
    return _tokenizer.__class__.__name__ in [
        "RobertaTokenizer",
        "LongformerTokenizer",
        "BartTokenizer",
        "RobertaTokenizerFast",
        "LongformerTokenizerFast",
        "BartTokenizerFast",
    ]


def get_sep_tokens(_tokenizer: PreTrainedTokenizer):
    return [_tokenizer.sep_token] * (_tokenizer.max_len_single_sentence - _tokenizer.max_len_sentences_pair)


def read_examples(file_path: str):
    data = json.load(open(file_path, 'r'))

    examples = []
    for sample in data:
        _context = sample["context"]
        _question = sample["question"]
        if "label" not in sample:
            _label = -1
        else:
            _label = sample["label"]
        examples.append({
            "context": _context,
            "question": _question,
            "options": sample["answers"],
            "label": _label
        })

    logger.info(f"{len(examples)} examples are loaded from {file_path}.")
    return examples


def _convert_example_to_features(example: Dict, tokenizer: PreTrainedTokenizer, max_seq_length: int) -> Dict:
    context = example["context"]
    question = example["question"]
    context_sentences = [sent for sent in sent_tokenize(context) if sent]

    context_tokens = []
    if is_bpe(tokenizer):
        specific_args = {"add_prefix_space": True}
    else:
        specific_args = {}
    for _sent_id, _sent in enumerate(context_sentences):
        if _sent_id > 0:
            _sent_tokens = tokenizer.tokenize(_sent, **specific_args)
        else:
            _sent_tokens = tokenizer.tokenize(_sent)
        context_tokens.extend([(_sent_id, _tok) for _tok in _sent_tokens])

    _q_sent_id_offset = len(context_sentences)
    question_tokens = [(_q_sent_id_offset, _tok) for _tok in tokenizer.tokenize(question)]

    features = []
    for option in example["options"]:
        sep_tokens = get_sep_tokens(tokenizer)
        _op_sent_id_offset = _q_sent_id_offset + 1
        opt_tokens = [(_op_sent_id_offset, _tok) for _tok in tokenizer.tokenize(option)]

        lens_to_remove = len(context_tokens) + len(question_tokens) + len(opt_tokens) + len(sep_tokens) + (
                tokenizer.model_max_length - tokenizer.max_len_sentences_pair) - max_seq_length

        tru_c_tokens, tru_q_o_tokens, _ = tokenizer.truncate_sequences(context_tokens,
                                                                       question_tokens + sep_tokens + opt_tokens,
                                                                       num_tokens_to_remove=lens_to_remove,
                                                                       truncation_strategy=TruncationStrategy.ONLY_FIRST)

        c_tokens, q_op_tokens = [], []
        sent_id_map = Counter()

        for _sent_id, _tok in tru_c_tokens:
            sent_id_map[_sent_id] += 1
            c_tokens.append(_tok)

        for _tok in tru_q_o_tokens:
            if isinstance(_tok, tuple):
                _sent_id, _tok = _tok
                q_op_tokens.append(_tok)
                sent_id_map[_sent_id] += 1
            elif isinstance(_tok, str):
                q_op_tokens.append(_tok)
            else:
                raise RuntimeError(_tok)

        sent_span_offset = 1  # [CLS]
        sent_spans = []
        for i in range(len(context_sentences) + 2):
            if i == _q_sent_id_offset or i == _op_sent_id_offset:
                sent_span_offset += (tokenizer.max_len_single_sentence - tokenizer.max_len_sentences_pair)  # [SEP]
            if i in sent_id_map:
                _cur_len = sent_id_map.pop(i)
                sent_spans.append((sent_span_offset, sent_span_offset + _cur_len))
                sent_span_offset += _cur_len
        assert not sent_id_map

        tokenizer_outputs = tokenizer(tokenizer.convert_tokens_to_string(c_tokens),
                                      text_pair=tokenizer.convert_tokens_to_string(q_op_tokens),
                                      padding=PaddingStrategy.MAX_LENGTH,
                                      max_length=max_seq_length)
        assert len(tokenizer_outputs["input_ids"]) == max_seq_length, (len(c_tokens), len(q_op_tokens), len(tokenizer_outputs["input_ids"]))
        features.append({
            "input_ids": tokenizer_outputs["input_ids"],
            "attention_mask": tokenizer_outputs["attention_mask"],
            "token_type_ids": tokenizer_outputs["token_type_ids"] if "token_type_ids" in tokenizer_outputs else None,
            "sentence_spans": sent_spans,
        })
    return {
        "features": features,
        "label": example["label"]
    }


def _data_to_tensors(features: List[Dict]):
    data_num = len(features)
    option_num = len(features[0]["features"])
    assert option_num == 4
    max_seq_length = len(features[0]["features"][0]["input_ids"])

    input_ids = torch.tensor([[op["input_ids"] for op in f["features"]] for f in features])
    attention_mask = torch.tensor([[op["attention_mask"] for op in f["features"]] for f in features], dtype=torch.long)
    if "token_type_ids" in features[0]["features"][0]:
        token_type_ids = torch.tensor([[op["token_type_ids"] for op in f["features"]] for f in features], dtype=torch.long)
    else:
        token_type_ids = None
    labels = torch.tensor([f["label"] for f in features], dtype=torch.long)

    # List[List[List[Tuple[int, int]]]]
    sentence_spans_ls = [[op["sentence_spans"] for op in f["features"]] for f in features]
    max_sent_num = 0
    for f in sentence_spans_ls:
        f_max_sent_num = max(map(len, f))
        max_sent_num = max(f_max_sent_num, max_sent_num)

    sentence_spans = torch.zeros(data_num, option_num, max_sent_num, 2, dtype=torch.long).fill_(-1)
    for f_id, f in enumerate(sentence_spans_ls):
        for op_id, op in enumerate(f):
            f_op_sent_num = len(op)
            sentence_spans[f_id, op_id, :f_op_sent_num] = torch.tensor(op, dtype=torch.long)

    if token_type_ids is not None:
        return input_ids, attention_mask, token_type_ids, labels, sentence_spans
    else:
        return input_ids, attention_mask, labels, sentence_spans


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, num_workers: int = 16):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{max_seq_length}"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        examples, features, tensors = torch.load(cached_file_path)
        return examples, features, tensors

    examples = read_examples(file_path)

    with Pool(num_workers) as p:
        _annotate = partial(_convert_example_to_features, tokenizer=tokenizer, max_seq_length=max_seq_length)
        features = list(tqdm(
            p.imap(_annotate, examples, chunksize=32),
            total=len(examples),
            desc='converting examples to features:'
        ))

    logger.info("Transform features into tensors...")
    tensors = _data_to_tensors(features)

    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((examples, features, tensors), cached_file_path)

    return examples, features, tensors
