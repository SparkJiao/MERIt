import copy
import os
import pickle
import random
from typing import Dict, List
from multiprocessing import Pool
from functools import partial

import torch
from torch.distributions.geometric import Geometric
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from datasets.wiki_entity_path_v5 import WikiPathDatasetV5
from general_util.logger import get_child_logger

logger = get_child_logger("Wiki.Entity.Path.V5.1")

_entity_pool: Dict
_all_neg_candidates: List
_geometric_dist: torch.distributions.Distribution


def replace_ent_neg_double(candidate, h_ent_id, h_ent_str, t_ent_id, t_ent_str, rep_pairs: Dict[int, str] = None,
                           out_of_domain: bool = False):
    # If the negative candidate comes from other data item, the entity id is not compatible.
    # So
    #   1. The head and tail entity id should not be considered to be contained in the negative samples.
    #   2. Empty the ``rep_pairs`` since the entity is not compatible. But keep the head or tail entity if should be replaced.

    entities = candidate["ent"]
    if rep_pairs is not None:
        if out_of_domain:
            _rep_pairs_copy = {}
            if h_ent_id in rep_pairs:
                _rep_pairs_copy[h_ent_id] = rep_pairs[h_ent_id]
            if t_ent_id in rep_pairs:
                _rep_pairs_copy[t_ent_id] = rep_pairs[t_ent_id]
        else:
            _rep_pairs_copy = copy.deepcopy(rep_pairs)  # Avoid modification on the original pairs dict.
    else:
        _rep_pairs_copy = {}
    id2ent = {ent_id: ent.values() for ent_id, ent in entities.items()}

    if out_of_domain:
        filtered_entities = list(entities.keys())
        h_t_entities = []
    else:
        filtered_entities = [ent_id for ent_id in entities if ent_id not in [h_ent_id, t_ent_id]]
        h_t_entities = []
        for _tmp in [h_ent_id, t_ent_id]:
            if _tmp in entities:
                h_t_entities.append(_tmp)
        assert len(h_t_entities) < 2

    if len(filtered_entities) == 0:
        return None
    else:
        # If there is already an head/tail entity in the negative sentence, we only sample another entity to replace.
        if len(h_t_entities) == 1:
            re1 = h_t_entities[0]
            re2 = random.choice(filtered_entities)
        else:
            re1, re2 = random.sample(filtered_entities, 2)
    re = [re1, re2]
    tgt = [h_ent_str, t_ent_str]

    _same_n = 0
    for _tmp in tgt:
        for _tmp_re in re:
            _all_ent_mention_tmp = set([_tmp_ent_mention["name"] for _tmp_ent_mention in id2ent[_tmp_re]])
            # for _tmp_ent_mention in id2ent[_tmp_re]:
            if _tmp in _all_ent_mention_tmp:
                _same_n += 1
                break
    if _same_n >= 2:
        logger.warning(f"Same replacement found. ({tgt}, {re[0]['name']}, {re[1]['name']}) Continue.")
        return None

    # If the head/tail entity will be replaced for augmentation,
    # replace the string for replacement for negative sample construction with the target string directly
    if _rep_pairs_copy:
        if h_ent_id in _rep_pairs_copy:
            h_ent_str = _rep_pairs_copy[h_ent_id]
        if t_ent_id in _rep_pairs_copy:
            t_ent_str = _rep_pairs_copy[t_ent_id]
        tgt = [h_ent_str, t_ent_str]

    random.shuffle(tgt)
    if out_of_domain:
        _rep_pairs_copy.clear()  # After get the modified head/tail entity string. Empty it since the entity is not compatible.
    _rep_pairs_copy[re[0]] = tgt[0]
    _rep_pairs_copy[re[1]] = tgt[1]

    return _replace_entities_w_str(candidate, _rep_pairs_copy)


def _replace_entities_w_str(candidate, rep_pairs: Dict[int, str]):
    ent_to_rep = []

    ent_vis = set()

    for ent_id in candidate["ent"]:
        if ent_id in rep_pairs:
            for r in candidate["ent"][ent_id].values():
                r["tgt"] = rep_pairs[ent_id]
                ent_to_rep.append(r)
            assert ent_id not in ent_vis
            ent_vis.add(ent_id)

    re = sorted(ent_to_rep, key=lambda x: x["pos"][0])
    # Non-overlapping check.
    for _tmp_id, _tmp in enumerate(re):
        if _tmp_id == 0:
            continue
        assert _tmp["pos"][0] >= re[_tmp_id - 1]["pos"][1]

    new_tokens = []
    _last_e = 0
    for r in re:
        s, e = r["pos"]
        new_tokens.extend(candidate["sent"][_last_e: s])
        new_tokens.append(r["tgt"])
        _last_e = e

    new_tokens.extend(candidate["sent"][_last_e:])
    return " ".join(new_tokens)


def replace_neg(pos_candidate, neg_candidate, rep_pairs: Dict[int, str] = None, out_of_domain: bool = False):
    # h = pos_candidate["h"]
    # assert h in pos_candidate["ent"], (pos_candidate["ent"].keys(), h)

    h_ls = pos_candidate["ent"][pos_candidate["h"]]
    h_ent = random.choice(list(h_ls.values()))
    h_ent_str = pos2str(h_ent["pos"][0], h_ent["pos"][1], pos_candidate["sent"])

    t_ls = pos_candidate["ent"][pos_candidate["t"]]
    t_ent = random.choice(list(t_ls.values()))
    t_ent_str = pos2str(t_ent["pos"][0], t_ent["pos"][1], pos_candidate["sent"])

    # if out_of_domain:
    # If the negative candidate comes from other data item, the entity id is not compatible.
    # So
    #   1. The head and tail entity id should not be considered to be contained in the negative samples.
    #   2. Empty the ``rep_pairs`` since the entity is not compatible. But keep the head or tail entity if should be replaced.
    _rep_res = replace_ent_neg_double(neg_candidate, pos_candidate["h"], h_ent_str, pos_candidate["t"], t_ent_str,
                                      rep_pairs=rep_pairs, out_of_domain=out_of_domain)
    # else:
    #     _rep_res = replace_ent_neg_double(neg_candidate, pos_candidate["h"], h_ent_str, pos_candidate["t"], t_ent_str,
    #                                       rep_pairs=rep_pairs, out_of_domain=out_of_domain)
    return _rep_res


def pos2str(ent_s, ent_e, tokens):
    return " ".join(tokens[ent_s: ent_e])


def sample_entity(pool, src_id, k):
    all_data_id_ls = list(pool.keys())

    res = []
    pool_vis = set()
    for _ in range(k):
        pool_id = random.choice(all_data_id_ls)
        while pool_id == src_id:
            pool_id = random.choice(all_data_id_ls)

        pool_vis.add(pool_id)

        entity_ls = list(pool[pool_id].values())  # all entities, each one contains a list of all positions.
        entity = random.choice(entity_ls)  # sample an entity.
        entity_str = random.choice(entity)  # sample an position (mention).
        res.append(entity_str)
    return res


def _initializer(entity_pool: Dict, all_neg_candidates: List, geometric_dist: torch.distributions.Distribution):
    global _entity_pool
    global _all_neg_candidates
    global _geometric_dist

    _entity_pool = entity_pool
    _all_neg_candidates = all_neg_candidates
    _geometric_dist = geometric_dist


def _process_single_item(item, max_neg_num: int, aug_num: int, min_rep_num: int, shuffle_context: bool):
    examples = []

    selected_sentences = item["selected_sentences"]
    if len(selected_sentences) == 0:
        return []
    context = " ".join([" ".join(s["sent"]) for s_id, s in selected_sentences.items()])

    neg_candidates = [x for x in item["rest_sentences"].values() if len(x["ent"]) > 1]

    for pos_idx, pos_candi in enumerate(item["pos"]):

        neg_res = []
        for neg in neg_candidates:
            _rep_res = replace_neg(pos_candi, neg, rep_pairs=None)
            if _rep_res is not None:
                neg_res.append(_rep_res)
                if len(neg_res) == max_neg_num:
                    break

        while len(neg_res) < max_neg_num:
            neg_data_id, neg = random.choice(_all_neg_candidates)
            while neg_data_id == item["id"]:
                neg_data_id, neg = random.choice(_all_neg_candidates)
            _rep_res = replace_neg(pos_candi, neg, rep_pairs=None, out_of_domain=True)
            if _rep_res is not None:
                neg_res.append(_rep_res)
                if len(neg_res) == max_neg_num:
                    break

        examples.append({
            "context": context,
            "negative": neg_res,
            "positive": " ".join(pos_candi["sent"]),
            "orig_id": item["id"]
        })

    # Augment the context
    # 1. ~~Choose the head entity or the tail entity as the target entity.~~
    #    Randomly sample some entities in the path to be replaced.
    # 2. Randomly sample other entities from the entity pool.
    # 3. Replace the target entity in the context with the sampled entity.
    # 4. Replace the target entity in negative samples with the sampled entity.

    # Gather the entity ids in the meta-path for sampling.
    path_ent_ids = set([p_ent_id for p_ent_id, p_sent_id in item["path"]])
    h_t_ent_ids = [item["pos"][0]["h"], item["pos"][0]["t"]]
    for x in h_t_ent_ids:
        assert x in path_ent_ids
        path_ent_ids.remove(x)

    for _ in range(aug_num):  # Repeat for augmentation

        # Sample the amount of entities to be replaced from the geometric distribution.
        _sampled_ent_num = int(_geometric_dist.sample().item()) + min_rep_num
        cnt = 0
        while _sampled_ent_num >= (len(path_ent_ids) + len(h_t_ent_ids)):
            cnt += 1
            _sampled_ent_num = int(_geometric_dist.sample().item()) + min_rep_num
            if cnt > 1000:
                logger.warning("Wrong here.")
                raise RuntimeError()
        assert min_rep_num <= _sampled_ent_num < (len(path_ent_ids) + len(h_t_ent_ids))

        # Make sure the head/tail entity in the entities to be replaced.
        if _sampled_ent_num <= 2:
            sampled_ent_ids = random.sample(h_t_ent_ids, _sampled_ent_num)
        else:
            sampled_ent_ids = h_t_ent_ids + random.sample(list(path_ent_ids), _sampled_ent_num - 2)

        # TODO: How to check there is no repetition in the sampled entity strings ?
        target_ent_str = sample_entity(_entity_pool, item["id"], _sampled_ent_num)
        sampled_rep_pairs = {_ent_id: _rep_str for _ent_id, _rep_str in zip(sampled_ent_ids, target_ent_str)}

        new_sentences = []
        for _, sent in selected_sentences.items():
            new_sentences.append(_replace_entities_w_str(sent, sampled_rep_pairs))

        for pos_idx, pos_candi in enumerate(item["pos"]):
            new_pos_candi_sent = _replace_entities_w_str(pos_candi, sampled_rep_pairs)

            neg_res = []

            for neg in neg_candidates:
                _rep_res = replace_neg(pos_candi, neg, rep_pairs=sampled_rep_pairs)
                if _rep_res is not None:
                    neg_res.append(_rep_res)
                    if len(neg_res) == max_neg_num:
                        break

            while len(neg_res) < max_neg_num:
                # neg = random.choice(all_neg_candidates)
                neg_data_id, neg = random.choice(_all_neg_candidates)
                while neg_data_id == item["id"]:
                    neg_data_id, neg = random.choice(_all_neg_candidates)
                _rep_res = replace_neg(pos_candi, neg, rep_pairs=sampled_rep_pairs, out_of_domain=True)
                assert _rep_res is not None
                if _rep_res is not None:
                    neg_res.append(_rep_res)

            if shuffle_context:
                random.shuffle(new_sentences)
            new_context = " ".join(new_sentences)

            examples.append({
                "context": new_context,
                "negative": neg_res,
                "positive": new_pos_candi_sent,
                "orig_id": item["id"]
            })

    return examples


def read_examples(file_path: str, shuffle_context: bool = False,
                  max_neg_num: int = 3, aug_num: int = 10,
                  geo_p: float = 0.5, min_rep_num: int = 1, num_workers: int = 48):
    logger.info(f"Loading raw examples from {file_path}...")
    # raw_data = json.load(open(file_path, 'r'))
    raw_data = pickle.load(open(file_path, "rb"))
    data = raw_data["examples"]
    raw_texts = raw_data["raw_texts"]

    geometric_dist = Geometric(torch.tensor([geo_p]))

    all_neg_candidates = []
    for x in data:
        all_neg_candidates.extend([(x["id"], y) for y in x["rest_sentences"].values() if len(y["ent"]) > 1])

    entity_pool = {}  # example_id -> entity_id -> position list
    for x in tqdm(data, desc="Preparing entity pool", total=len(data)):
        if x["id"] in entity_pool:
            continue
        # x["entity"] -> Dict[ent_id, List[position]]
        all_str = {}
        for ent_id, ent_pos_ls in x["entity"].items():
            all_str[ent_id] = [
                pos2str(
                    _e_pos["pos"][0], _e_pos["pos"][1], x["all_sentences"][_e_pos["sent_id"]]
                ) for _e_pos in ent_pos_ls
            ]
        entity_pool[x["id"]] = all_str  # Dict[ent_id, List[str]]

    examples = []
    with Pool(num_workers, initializer=_initializer, initargs=(entity_pool, all_neg_candidates, geometric_dist)) as p:
        _annotate = partial(_process_single_item,
                            max_neg_num=max_neg_num, aug_num=aug_num, min_rep_num=min_rep_num, shuffle_context=shuffle_context)
        _results = list(tqdm(
            p.imap(_annotate, data, chunksize=32),
            total=len(data),
            desc="Reading examples"
        ))

    for _res in _results:
        if _res:
            examples.extend(_res)

    logger.info(f"{len(examples)} examples are loaded from {file_path}.")

    return examples, raw_texts


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer,
                                   shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
                                   max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
                                   num_workers=48):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
                  f"{max_seq_length}_{geo_p}_{min_rep_num}_path_v5_1"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        examples, raw_texts = torch.load(cached_file_path)
        dataset = WikiPathDatasetV5(examples, raw_texts)
        return examples, None, dataset

    examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num, aug_num=aug_num,
                                        geo_p=geo_p, min_rep_num=min_rep_num, num_workers=num_workers)

    # Save
    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((examples, raw_texts), cached_file_path)

    return examples, None, WikiPathDatasetV5(examples, raw_texts)
