import json
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import Dict, List
from copy import deepcopy

import torch
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from dataset.data_utils import get_sep_tokens, is_bpe
from general_util.logger import get_child_logger

logger = get_child_logger("ReClor.Sentence")


def replace_phrase_with_mask(text: str, phrase: str, tokenizer: PreTrainedTokenizer, mask_all: bool = False, use_unk: bool = False):
    rep_token = tokenizer.mask_token if not use_unk else tokenizer.unk_token
    idx = text.find(phrase)
    assert idx != -1, (text, phrase)
    if mask_all:
        pass
    else:
        num_tokens = len(tokenizer.tokenize(phrase))
        mask = " ".join([tokenizer.mask_token] * num_tokens)
        return text[:idx] + mask + text[(idx + len(phrase)):]


def read_examples(file_path: str, tokenizer: PreTrainedTokenizer):
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
        gt_id = len(examples) - 1
        if 'mask_phrase' in sample:
            context_phrase = sample['mask_phrase'][0]
            answer_phrase = sample['mask_phrase'][1]
            options = deepcopy(sample["answers"])
            options[_label] = replace_phrase_with_mask(options[_label], answer_phrase, tokenizer)

            if 'mask_ops' in sample:
                for op_id, _phrase in sample['mask_ops']:
                    assert op_id != _label
                    options[op_id] = replace_phrase_with_mask(options[op_id], _phrase, tokenizer)

            examples.append({
                "context": replace_phrase_with_mask(_context, context_phrase, tokenizer),
                "question": _question,
                "options": options,
                "label": _label,
                "gt_id": gt_id
            })
        elif 'mask_ops' in sample:
            for context_np, op_ops in sample['mask_ops']:
                mask_context = replace_phrase_with_mask(_context, context_np, tokenizer)
                options = deepcopy(sample["answers"])
                # print(op_ops)
                for op_id, op_np in op_ops:
                    options[op_id] = replace_phrase_with_mask(options[op_id], op_np, tokenizer)
                examples.append({
                    "context": mask_context,
                    "question": _question,
                    "options": options,
                    "label": _label,
                    "gt_id": gt_id
                })

    logger.info(f"{len(examples)} examples are loaded from {file_path}.")
    return examples


def _convert_example_to_features(example: Dict, tokenizer: PreTrainedTokenizer, max_seq_length: int) -> Dict:
    context = example["context"]
    question = example["question"]
    context_sentences = [sent for sent in sent_tokenize(context) if sent]

    context_tokens = []
    for _sent_id, _sent in enumerate(context_sentences):
        if is_bpe(tokenizer):
            _sent = " " + _sent
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
    if "gt_id" in example:
        return {
            "features": features,
            "label": example["label"],
            "gt_id": example["gt_id"]
        }
    return {
        "features": features,
        "label": example["label"]
    }


def _data_to_tensors(features: List[Dict], add_mlm: bool = False):
    data_num = len(features)
    option_num = len(features[0]["features"])
    assert option_num == 4
    max_seq_length = len(features[0]["features"][0]["input_ids"])

    input_ids = torch.tensor([[op["input_ids"] for op in f["features"]] for f in features])
    attention_mask = torch.tensor([[op["attention_mask"] for op in f["features"]] for f in features], dtype=torch.long)
    if features[0]["features"][0]["token_type_ids"] is not None:
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

    # Generate MLM labels
    mlm_labels = torch.zeros(input_ids.size(), dtype=torch.long).fill_(-1)
    for f_id, f in enumerate(features):
        if "gt_id" in f:
            assert f["gt_id"] < f_id
            for op_id, op_input_ids in enumerate(input_ids[f_id]):
                op_mlm_mask = op_input_ids != input_ids[f["gt_id"], op_id]
                if op_mlm_mask.sum().item() >= 20:
                    logger.info("Inconsistency found. Continue")
                    continue
                mlm_labels[f_id, op_id] = input_ids[f["gt_id"], op_id].masked_fill(~op_mlm_mask, -1)

    if token_type_ids is not None:
        res = (input_ids, attention_mask, token_type_ids, labels, sentence_spans,)
    else:
        res = (input_ids, attention_mask, labels, sentence_spans,)

    if add_mlm:
        res = res + (mlm_labels,)

    return res


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, add_mlm: bool = False,
                                   num_workers: int = 16):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{max_seq_length}_{'w_mlm_' if add_mlm else ''}mask"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        examples, features, tensors = torch.load(cached_file_path)
        return examples, features, tensors

    examples = read_examples(file_path, tokenizer)

    with Pool(num_workers) as p:
        _annotate = partial(_convert_example_to_features, tokenizer=tokenizer, max_seq_length=max_seq_length)
        features = list(tqdm(
            p.imap(_annotate, examples, chunksize=32),
            total=len(examples),
            desc='converting examples to features:'
        ))

    logger.info("Transform features into tensors...")
    tensors = _data_to_tensors(features, add_mlm=add_mlm)

    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((examples, features, tensors), cached_file_path)

    return examples, features, tensors
