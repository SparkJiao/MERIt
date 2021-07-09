import glob
import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

"""
Process the data from ERICA pretrain: ERICA: 
    Improving Entity and Relation Understanding for Pre-trained Language Models via Contrastive Learning
"""

_tokenizer: PreTrainedTokenizer
ENT_1 = '[ENT_1]'
ENT_2 = '[ENT_2]'


def _initializer(ini_tokenizer: PreTrainedTokenizer):
    global _tokenizer
    _tokenizer = ini_tokenizer

    _tokenizer.add_tokens(["[ENT_1]", "[ENT_2]"])


def process_entity_single_sent(_sentence: List[str], ent, ent_str):
    _s, _e = ent['pos']
    _rep_sent = _sentence[:_s] + [ent_str] * (_e - _s) + _sentence[_e:]
    return _rep_sent


def process_entity(sentences: List[List[str]], ent, ent_str):
    for _e in ent:
        _s_id = _e['sent_id']
        sentences[_s_id] = process_entity_single_sent(sentences[_s_id], _e, ent_str)


def union_special_token(_sentence: List[str]):
    _new_sent = []
    idx = 0
    while idx < len(_sentence):
        tk = _sentence[idx]
        if tk not in [ENT_1, ENT_2]:
            _new_sent.append(tk)
            idx += 1
        else:
            _new_sent.append(tk)
            while idx < len(_sentence) and _sentence[idx] == tk:
                idx += 1
    return _new_sent


def process_sentences(sentences: List[List[str]], ent_1, ent_2):
    sentences = deepcopy(sentences)

    process_entity(sentences, ent_1, ENT_1)
    process_entity(sentences, ent_2, ENT_2)

    new_sentences = []
    for sent in sentences:
        new_sentences.append(union_special_token(sent))
    return new_sentences


def process_example(_example, sentences: List[List[str]], entities):
    pos_sent = _example['pos']
    neg_sent = _example['neg']
    h = entities[_example['h']]
    t = entities[_example['t']]

    # filtered_sentences = [s for s_id, s in enumerate(sentences) if s_id not in [pos_sent, neg_sent]]
    all_processed_sentences = process_sentences(sentences, h, t)

    rep_filtered_sentences = [s for s_id, s in enumerate(all_processed_sentences) if s_id not in [pos_sent, neg_sent]]

    _pos_sub_h = _neg_sub_h = None
    _pos_sub_t = _neg_sub_t = None
    for _sub_h in h:
        if _pos_sub_h is None and _sub_h['sent_id'] == pos_sent:
            _pos_sub_h = _sub_h
        if _neg_sub_h is None and _sub_h['sent_id'] == neg_sent:
            _neg_sub_h = _sub_h

    for _sub_t in t:
        if _pos_sub_t is None and _sub_t['sent_id'] == pos_sent:
            _pos_sub_t = _sub_t
        if _neg_sub_t is None and _sub_t['sent_id'] == neg_sent:
            _neg_sub_t = _sub_t

    assert _pos_sub_h and _pos_sub_t
    if neg_sent != -1:
        assert _neg_sub_h and _neg_sub_t

    pos_sent = process_entity_single_sent(sentences[pos_sent], _pos_sub_h, ENT_1)
    pos_sent = union_special_token(process_entity_single_sent(pos_sent, _pos_sub_t, ENT_2))

    if neg_sent != -1:
        # Switch the Entity Symbol to construct a negative sample.
        neg_sent = process_entity_single_sent(sentences[neg_sent], _neg_sub_h, ENT_2)
        neg_sent = union_special_token(process_entity_single_sent(neg_sent, _neg_sub_t, ENT_1))
    else:
        neg_sent = None

    return {
        "sentences": rep_filtered_sentences,
        "pos": pos_sent,
        "neg": neg_sent
    }


def construct_sample(sample):
    entities = sample["vertexSet"]

    relations = sample["labels"]

    sentences = sample["sents"]

    edges = defaultdict(list)
    for rel in relations:
        edges[rel['h']].append(rel['t'])

    relations = {(item['h'], item['t']): item['r'] for item in relations}

    # sent2ent = defaultdict(list)
    # for ent_id, ent in enumerate(entities):
    #     sent2ent[ent['sent_id']].append(ent_id)

    examples = []
    for (h, t) in relations.keys():
        h_sent_cnt = len(entities[h])
        t_sent_cnt = len(entities[t])

        if h_sent_cnt == 1 and t_sent_cnt == 1:
            continue

        h_sent_ids = set([_e['sent_id'] for _e in entities[h]])
        t_sent_ids = set([_e['sent_id'] for _e in entities[t]])
        commons = list(h_sent_ids & t_sent_ids)
        if len(commons) >= 2:
            examples.append({
                'pos': commons[0],
                'neg': commons[1],
                'h': h,
                't': t
            })
        elif len(commons) == 1:
            examples.append({
                'pos': commons[0],
                'neg': -1,
                'h': h,
                't': t
            })
        else:
            continue

    examples = [process_example(example, sentences, entities) for example in examples]

    return examples


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='roberta-base')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()

    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = list(glob.glob(args.input_file))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f'.{tokenizer_name}_path.json'

    for _file in input_files:
        samples = json.load(open(_file))

        processed_samples = []
        with Pool(args.num_workers, initializer=_initializer, initargs=(tokenizer,)) as p:
            _annotate = partial(construct_sample)
            _results = list(tqdm(
                p.imap(_annotate, samples, chunksize=32),
                total=len(samples),
                desc='processing samples'
            ))

        neg_num = 0
        for res in _results:
            if res:
                processed_samples.extend(res)
                for e in res:
                    if e["neg"] is not None:
                        neg_num += 1

        print(f"Hard negative samples: {neg_num} / {len(processed_samples)} = {neg_num * 1.0 / len(processed_samples)}.")

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            _file_name = _file.split('/')[-1]
            output_file = os.path.join(args.output_dir, _file_name.replace('.json', file_suffix))
            json.dump(processed_samples, open(output_file, 'w'))
        else:
            output_file = _file.replace('.json', file_suffix)
            json.dump(processed_samples, open(output_file, 'w'))

    print("Done.")

