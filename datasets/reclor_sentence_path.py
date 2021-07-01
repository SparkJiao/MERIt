import json
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import torch
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from datasets.data_utils import get_sep_tokens, is_bpe
from general_util.logger import get_child_logger

logger = get_child_logger("ReClor.Sentence.Path")


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
            "path": sample["sub_sentence_ids"] if "sub_sentence_ids" in sample else None,
            "label": _label
        })

    logger.info(f"{len(examples)} examples are loaded from {file_path}.")
    return examples


def _convert_example_to_features(example: Dict, tokenizer: PreTrainedTokenizer, max_seq_length: int,
                                 include_q: bool = True, include_op: bool = True) -> Dict:
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
    enhanced_path_num = 0
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

        # If there is truncation, use the map for validation check.
        _truncated_c_sen_num = len(sent_id_map)
        _c_sent_orig2now_map = {}
        for now_id, orig_id in enumerate(sent_id_map.keys()):
            _c_sent_orig2now_map[orig_id] = now_id

        for _tok in tru_q_o_tokens:
            if isinstance(_tok, tuple):
                _sent_id, _tok = _tok
                q_op_tokens.append(_tok)
                sent_id_map[_sent_id] += 1
            elif isinstance(_tok, str):
                q_op_tokens.append(_tok)
            else:
                raise RuntimeError(_tok)

        # Add question and option mapping into the map.
        assert _q_sent_id_offset not in _c_sent_orig2now_map
        _c_sent_orig2now_map[_q_sent_id_offset] = _truncated_c_sen_num
        assert _op_sent_id_offset not in _c_sent_orig2now_map
        _c_sent_orig2now_map[_op_sent_id_offset] = _truncated_c_sen_num + 1

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

        if example["path"] is not None:
            path = example["path"]
        else:
            path = list(range(len(context_sentences)))
        if include_q:
            path += [_q_sent_id_offset]
        if include_op:
            path += [_op_sent_id_offset]
        path = [_c_sent_orig2now_map[x] for x in path if x in _c_sent_orig2now_map]
        rev_path = [x for x in range(len(sent_spans)) if x not in path]

        if len(rev_path) > 0:
            enhanced_path_num += 1

        features.append({
            "input_ids": tokenizer_outputs["input_ids"],
            "attention_mask": tokenizer_outputs["attention_mask"],
            "token_type_ids": tokenizer_outputs["token_type_ids"] if "token_type_ids" in tokenizer_outputs else None,
            "sentence_spans": sent_spans,
            "path": path,
            "rev_path": rev_path
        })
    return {
        "features": features,
        "label": example["label"],
        "enhanced": enhanced_path_num > 0
    }


def _data_to_tensors(features: List[Dict]):
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

    path_ls = [[op["path"] for op in f["features"]] for f in features]
    rev_path_ls = [[op["rev_path"] for op in f["features"]] for f in features]

    max_path_len = 0
    max_rev_path_len = 0
    for f_path, f_rev_path in zip(path_ls, rev_path_ls):
        f_max_path_len = max(map(len, f_path))
        f_max_rev_path_len = max(map(len, f_rev_path))
        max_path_len = max(max_path_len, f_max_path_len)
        max_rev_path_len = max(max_rev_path_len, f_max_rev_path_len)

    path = torch.zeros(data_num, option_num, max_path_len, dtype=torch.long).fill_(-1)
    rev_path = torch.zeros(data_num, option_num, max_rev_path_len, dtype=torch.long).fill_(-1)
    for f_id, (f_path, f_rev_path) in enumerate(zip(path_ls, rev_path_ls)):
        assert len(f_path) == len(f_rev_path)
        for op_id in range(len(f_path)):
            _path_len = len(f_path[op_id])
            _rev_path_len = len(f_rev_path[op_id])
            path[f_id, op_id, :_path_len] = torch.tensor(f_path[op_id], dtype=torch.long)
            rev_path[f_id, op_id, :_rev_path_len] = torch.tensor(f_rev_path[op_id], dtype=torch.long)

    logger.info(f"Size of ``sentence_spans``: {sentence_spans.size()}")
    logger.info(f"Size of ``path``: {path.size()}")
    logger.info(f"Size of ``rev_path``L: {rev_path.size()}")

    if token_type_ids is not None:
        return input_ids, attention_mask, token_type_ids, labels, sentence_spans, path, rev_path
    else:
        return input_ids, attention_mask, labels, sentence_spans, path, rev_path


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, num_workers: int = 16,
                                   include_q: bool = True, include_op: bool = True):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{max_seq_length}{'_q' if not include_q else ''}{'_op' if not include_op else ''}_w.path"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        examples, features, tensors = torch.load(cached_file_path)
        return examples, features, tensors

    examples = read_examples(file_path)

    with Pool(num_workers) as p:
        _annotate = partial(_convert_example_to_features, tokenizer=tokenizer, max_seq_length=max_seq_length,
                            include_q=include_q, include_op=include_op)
        features = list(tqdm(
            p.imap(_annotate, examples, chunksize=32),
            total=len(examples),
            desc='converting examples to features:'
        ))

    enhanced = 0
    for f in features:
        _e = f.pop("enhanced")
        if _e:
            enhanced += 1
    logger.info(f"Path enhanced samples: {enhanced} / {len(features)}")

    logger.info("Transform features into tensors...")
    tensors = _data_to_tensors(features)

    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((examples, features, tensors), cached_file_path)

    return examples, features, tensors
