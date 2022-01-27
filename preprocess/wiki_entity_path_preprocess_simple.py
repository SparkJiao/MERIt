import copy
import glob
import json
import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Tuple, Dict, Set

from tqdm import tqdm

"""
Process the data from ERICA pretrain: ERICA: 
    Improving Entity and Relation Understanding for Pre-trained Language Models via Contrastive Learning

Core:
    Given a meta-path: <e_{a,i}, r_{i,j}, e_{a,j}>, ..., <e_{b,j}, r_{j,k}, e_{b,k}>,
    in which e_{i,j} represents the mention of the j-th entity in the i-th sentence,
    if there is a annotated relation between e_i and e_k, there is a logic path exists.
    We extract the path as a positive example.
    For negative samples, ...

Procedure:
    1.  枚举relation，判断是否存在一条非直接的path连接relation的头实体和尾实体。路径的游走要求：
        对于一对实体<e_i, e_j>:
        1.  存在relation连接<e_i, e_j>
        2.  <e_i, e_j>在同一个句子里。
        可选的，可以试试只用1或者只用2。 其中e_i和e_j都是可以跨句的（common mention）

Version 4.1 Update:
    1.  不再枚举relation，枚举<e_i, e_j>要求 e_i 和 e_j 只要出现在同一个句子里即可。

Version 5.0 Update:
    1.  实体保存形式从Tuple改成Dict方便对实体整体进行sample
    2.  修改bug：``ent2sent``和``sent2ent``的从list改成用set保存
    3.  修改bug: 上个版本中没有区分同一个实体在同一个句子里的不同位置上mention，
        这可能会导致对于某个需要被替换的实体只替换了其中一个mention，而另一个mention被保留

Version 6.0 Update:
    1.  对于meta-path上的每一条边记录这条边上的头实体和尾实体，与``pos``等同对待。
    2.  注释掉了Line 80，因为通过远程关系的边连接的两个实体不在同一个句子里（如果在同一个句子里会优先通过 path 1 找到），
        此时该句子可能包含别的实体，所以依然是可以延着这个句子搜索的，即再执行一次共句的实体搜索。

        此时会有一个现象，如果在``path``里存在一个句子的记录，该记录只包含头实体或者尾实体，此时该实体一定是一个
        通过远程关系连接的实体，此时我们假设了该句子的内容对path没有影响，即没有对逻辑关系进行了描述的贡献，
        因为对逻辑关系的描述隐含在了潜在的relation当中，而不是句子内容。

        带来的后果是我们其实不应当对这个实体进行实体替换，因为一旦替换，潜在的relation不再存在，逻辑关系也不再存在。
        如果严谨的话这里应该考虑对原有的context实体的替换逻辑进行修正，即首先筛选出这些孤立实体（以及参与了远程relation的实体？），
        只能在剩下的实体中进行替换。（或者直接去掉通过潜在relation连接的实体？）
Version 7.0 Update:
    TODO：
    1.  Version 6 做了一个假设，也就是每个句子只能用来连接两个实体，不能被重复使用
        这其实会导致数据减少，举例：e_1 经由 s_1 找到了 e_2，然后 s_1 被去掉，不能再用，如果只有 s_2 中包含了 e_3，
        那么此次搜索不可能再找到 e_3，因为 s_2 已经被拿来连接 e_1 和 e_2 了，但实际上是可以访问 e_3 的
        是否需要修复？
        
        好像其实是不需要修复的，因为在这个逻辑下，e_1 -> s_1 -> e_3 也是会被访问到的， 后续节点只要和 e_2 / e_3 中的任意一个相连就能被访问到。
    2.  ~~是否修复Version 6.0 Update 2 提到的这一点，加一个数组描述这个relation前后的实体不能被替换~~
    3.  ~~修复掉FIXME~~

Version Simple:
    Remove meta-path. Just select some sentence randomly.
"""


def extract_entities_of_sent(sent_id, sent2ent, entities) -> Dict[int, Dict]:
    ent_ls = defaultdict(dict)
    for e_id in sent2ent[sent_id]:
        for pos_id, e in enumerate(entities[e_id]):
            if e["sent_id"] == sent_id:
                ent_ls[e_id][pos_id] = e
    return ent_ls


def workflow(sample):
    entities = sample["vertexSet"]
    relations = sample["labels"]
    sentences = sample["sents"]

    rel_edges = defaultdict(dict)
    for item in relations:
        rel_edges[item["h"]][item["t"]] = item["r"]

    ent2sent = defaultdict(set)
    for ent_id, ent in enumerate(entities):
        for e in ent:
            ent2sent[ent_id].add(e["sent_id"])

    sent2ent = defaultdict(set)
    for ent_id, ent in enumerate(entities):
        for e in ent:
            sent2ent[e["sent_id"]].add(ent_id)

    # Enumerate over each relation, try to find a path starting from the head entity
    # and ending with the tail entity.
    # Version 4.1: Enumerate over each entity pair <e_i, e_j> such that e_i and e_j has the common sentences.
    examples = []

    all_sent_ids = list(range(len(sentences)))
    _sub_len = int(len(all_sent_ids) * 0.1)
    # pos_ids = all_sent_ids[:_sub_len]
    # ctx_ids = all_sent_ids[_sub_len: (9 * _sub_len)]
    # res_ids = all_sent_ids[(9 * _sub_len):]
    pos_ids = []
    ctx_ids = all_sent_ids[:(9 * _sub_len)]
    res_ids = all_sent_ids[(9 * _sub_len):]
    pos = [
        {
            "sent": sentences[_pos_id],
            "h": -1,
            "t": -1,
            "ent": extract_entities_of_sent(_pos_id, sent2ent, entities)
        } for _pos_id in pos_ids
    ]
    context = {
        s_id: {
            "sent": sentences[s_id],
            "ent": extract_entities_of_sent(s_id, sent2ent, entities),
            "h": -1,
            "t": -1
        } for s_id in ctx_ids
    }
    rest_sentences = {
        s_id: {
            "sent": sentences[s_id],
            "ent": extract_entities_of_sent(s_id, sent2ent, entities)
        } for s_id in res_ids
    }
    examples.append({
        "selected_sentences": context,
        "pos": pos,
        "rest_sentences": rest_sentences,
        "relation_connect_ent": [],
        "entity": {ent_id: ent for ent_id, ent in enumerate(entities)},
        "all_sentences": sentences
    })

    return examples


def extract_raw_text(sample):
    sentences = [" ".join(tokens) for tokens in sample["sents"]]
    return " ".join(sentences)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sample', default=False, action='store_true')

    args = parser.parse_args()

    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = list(glob.glob(args.input_file))

    file_suffix = f'.path_simple.pkl'
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for _file in input_files:
        samples = json.load(open(_file))

        processed_samples = []
        with Pool(args.num_workers) as p:
            _annotate = partial(workflow)
            _results = list(tqdm(
                p.imap(_annotate, samples, chunksize=32),
                total=len(samples),
                desc='processing samples'
            ))

        no_negative_num = 0
        for ex_id, _res in enumerate(_results):
            if _res:
                for _r in _res:
                    _r["id"] = ex_id
                    processed_samples.append(_r)

        for ex in processed_samples:
            if len(ex["rest_sentences"]) == 0:
                no_negative_num += 1

        print(f"Processed examples: {len(processed_samples)}")
        print(f"Examples without hard negative samples: {no_negative_num}.")

        avg_sent_num = sum(map(lambda x: len(x["sents"]), samples)) / len(samples)
        print(f"Average sentence num: {avg_sent_num}")

        with Pool(args.num_workers) as p:
            _annotate = partial(extract_raw_text)
            _raw_texts = list(tqdm(
                p.imap(_annotate, samples, chunksize=32),
                total=len(samples),
                desc='extracting raw texts'
            ))

        raw_texts = [t for t in _raw_texts if t.strip()]
        print(f"Extracted {len(raw_texts)} segments of text.")

        if args.output_dir:
            _file_name = _file.split('/')[-1]
            output_file = os.path.join(args.output_dir, _file_name.replace('.json', file_suffix))
        else:
            output_file = _file.replace('.json', file_suffix)

        if args.sample and len(processed_samples) > 30:
            pickle.dump(processed_samples[:30], open(output_file.replace('.pkl', '.sample.pkl'), 'wb'))
            break

        pickle.dump({"examples": processed_samples, "raw_texts": raw_texts}, open(output_file, 'wb'))

    print("Done.")
