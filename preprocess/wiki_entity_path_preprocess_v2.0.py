import glob
import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple, Dict, Set

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

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
        可选的，可以试试只用1或者只用2。
"""


def dfs(e_id, e_pos_id, entities, sent2ent, rel_edges, tgt_e_id, vis: Set, ent_vis: Set, path: Tuple, path_len: int, tgt_sent_ids: Set):
    # TODO: Currently, no path can be find.
    src_e = entities[e_id][e_pos_id]
    for next_e_id, next_e_pos_id in sent2ent[src_e["sent_id"]]:
        if next_e_id == e_id:
            continue
        if next_e_id in ent_vis:
            continue
        if (next_e_id, next_e_pos_id) in vis:
            continue
        if entities[next_e_id][next_e_pos_id]["sent_id"] in tgt_sent_ids:
            continue
        if e_id in rel_edges and next_e_id in rel_edges[e_id]:
            if next_e_id == tgt_e_id and path_len > 1:
                return path + ((next_e_id, next_e_pos_id),)
            vis.add((next_e_id, next_e_pos_id))
            cur_ent_vis = deepcopy(ent_vis)
            cur_ent_vis.add(next_e_id)
            res = dfs(next_e_id, next_e_pos_id, entities, sent2ent, rel_edges, tgt_e_id, vis, cur_ent_vis,
                      path + ((next_e_id, next_e_pos_id),), path_len + 1, tgt_sent_ids)
            if res is not None:
                return res
    return None


def process_path(path: Tuple, sentences, entities, sent2ent):
    selected_sent_ids = set()
    new_path = []
    for e_id, e_pos_id in path:
        sent_id = entities[e_id][e_pos_id]["sent_id"]
        selected_sent_ids.add(sent_id)
        new_path.append(entities[e_id][e_pos_id])
    if len(selected_sent_ids) == len(sentences):
        return False, None, None, None

    selected_sent_ids = sorted(list(selected_sent_ids))
    selected_sentences = {s_id: {"sent": sentences[s_id], "ent": []} for s_id in selected_sent_ids}
    for e_id, e_pos_id in path:
        e = entities[e_id][e_pos_id]
        selected_sentences[e["sent_id"]]["ent"].append(e)

    # Extract the rest sentences and the contained entities to construct negative samples.
    rest_sentences = {s_id: {"sent": sentences[s_id], "ent": []} for s_id in range(len(sentences)) if s_id not in selected_sentences}
    for s_id in rest_sentences:
        for e_id, e_pos_id in sent2ent[s_id]:
            rest_sentences[s_id]["ent"].append(entities[e_id][e_pos_id])

    return True, selected_sentences, rest_sentences, new_path


def workflow(sample):
    entities = sample["vertexSet"]
    relations = sample["labels"]
    sentences = sample["sents"]
    na_rel = sample["na_triple"]

    rel_edges = defaultdict(dict)
    for item in relations:
        rel_edges[item["h"]][item["t"]] = item["r"]

    # for item in na_rel:
    #     rel_edges[item[0]][item[1]] = -1

    sent2ent = defaultdict(list)
    for ent_id, ent in enumerate(entities):
        for pos_id, e in enumerate(ent):
            sent2ent[e["sent_id"]].append((ent_id, pos_id))

    # Enumerate over each relation, try to find a path starting from the head entity
    # and ending with the tail entity.
    examples = []
    for item in relations:
        h, t = item["h"], item["t"]
        h_sent_ids = set([_e["sent_id"] for _e in entities[h]])
        t_sent_ids = set([_e["sent_id"] for _e in entities[t]])
        common_sent_ids = h_sent_ids & t_sent_ids
        if len(common_sent_ids) == 0:
            continue
        for h_pos_id in range(len(entities[h])):
            vis_set = {(h, h_pos_id)}
            res = dfs(h, h_pos_id, entities, sent2ent, rel_edges, t, vis_set, {h}, ((h, h_pos_id),), 1, common_sent_ids)
            if res is not None:
                # print("Find a path.")
                flag, selected, rest, path = process_path(res, sentences, entities, sent2ent)
                if not flag:
                    continue
                examples.append({
                    "selected_sentences": selected,
                    "rest_sentences": rest,
                    "path": path
                })
    return examples


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

    file_suffix = f'.path_v2.json'
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

        for _res in _results:
            if _res:
                processed_samples.extend(_res)

        print(f"Processed examples: {len(processed_samples)}")

        avg_sent_num = sum(map(lambda x: len(x["sents"]), samples)) / len(samples)
        print(f"Average sentence num: {avg_sent_num}")

        if args.output_dir:
            _file_name = _file.split('/')[-1]
            output_file = os.path.join(args.output_dir, _file_name.replace('.json', file_suffix))
        else:
            output_file = _file.replace('.json', file_suffix)

        if args.sample and len(processed_samples) > 30:
            json.dump(processed_samples[:30], open(output_file.replace('.json', '.sample.json'), 'w'))
            break

        json.dump(processed_samples, open(output_file, 'w'))

    print("Done.")
