import copy
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import torch
from torch.distributions.geometric import Geometric
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from datasets.data_utils import get_all_permutation
from datasets.wiki_entity_path_v5 import WikiPathDatasetV5, WikiPathDatasetCollator
from general_util.logger import get_child_logger

"""
Version 6.0:
    During augmentation the dataset, if some negative samples should be sampled from other data items, use the corresponding head/tail
    entities for replacement, so that at least the these negative samples are model fluent to remove the bias.
Version 7.0:
    1.  Prior to use the examples from the same item.
Version 7.1:
    1.  Fix some minors.
Version 8.0:
    1.  Add replacement of context sentence.
"""

logger = get_child_logger("Wiki.Entity.Path.V8.0")

_entity_pool: Dict
_negative_pool: Dict
_all_neg_candidates: Dict
_geometric_dist: torch.distributions.Distribution

_permutation_sample_num: int = 6


def _switch_replace_neg(candidate, h_ent_id, h_ent_str, t_ent_id, t_ent_str, rep_pairs: Dict[int, str] = None):
    """
    Enumerate all the possible triplets and replace.
    """
    entities = candidate["ent"]

    non_target_ent_ids = [ent_id for ent_id in entities if ent_id not in [h_ent_id, t_ent_id]]
    h_t_ent_ids = [h_ent_id, t_ent_id]
    assert h_ent_id in entities and t_ent_id in entities

    str_map = {
        h_ent_id: h_ent_str,
        t_ent_id: t_ent_str
    }

    ent_name_set = {
        ent_id: set([_mention["name"] for _mention in entities[ent_id].values()]) for ent_id in entities
    }
    if rep_pairs is None:
        rep_pairs = {}

    # Currently, we only sample exactly one non-target
    neg_res = []
    for _non_tgt in non_target_ent_ids:
        _non_tgt_str = get_ent_str(candidate, _non_tgt)
        str_map[_non_tgt] = _non_tgt_str

        _source = h_t_ent_ids + [_non_tgt]
        _target = h_t_ent_ids + [_non_tgt]

        # The ``get_all_permutation`` function ensure that the obtained permutations
        # are **all** not the same with the initial permutation.
        _all_perm = get_all_permutation(_target)

        if _permutation_sample_num < len(_all_perm):
            _perm_sample_ls = random.sample(_all_perm, _permutation_sample_num)
        else:
            _perm_sample_ls = _all_perm

        for _perm in _perm_sample_ls:
            assert len(_perm) == len(_source)
            assert _perm != _source
            _rep_pairs_copy = copy.deepcopy(rep_pairs)
            _same_n = 0
            for _src, _tgt in zip(_source, _perm):
                _rep_pairs_copy[_src] = rep_pairs[_tgt] if _tgt in rep_pairs else str_map[_tgt]
                if _rep_pairs_copy[_src].lower() == str_map[_src].lower() or _rep_pairs_copy[_src].lower() in ent_name_set[_src]:
                    _same_n += 1
            if _same_n == len(_source):
                continue
            neg_res.append(_replace_entities_w_str(candidate, _rep_pairs_copy))

    return neg_res


def replace_ent_neg_double(candidate, h_ent_id, h_ent_str, t_ent_id, t_ent_str, rep_pairs: Dict[int, str] = None,
                           out_of_domain: bool = False):
    # If the negative candidate comes from other data item, the entity id is not compatible.
    # So
    #   1. The head and tail entity id should not be considered to be contained in the negative samples.
    #   2. Empty the ``rep_pairs`` since the entity is not compatible. But keep the head or tail entity if should be replaced.

    if rep_pairs is not None and h_ent_id in rep_pairs:
        h_ent_str = rep_pairs[h_ent_id]
    if rep_pairs is not None and t_ent_id in rep_pairs:
        t_ent_str = rep_pairs[t_ent_id]

    entities = candidate["ent"]

    if rep_pairs is None or out_of_domain:
        rep_pairs = {}

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
        assert len(h_t_entities) < 2, (candidate["ent"], (h_ent_str, t_ent_str))

    if len(filtered_entities) == 0:
        return []

    _source_ls = []

    if len(h_t_entities) == 1:
        for tgt in filtered_entities:
            _source_ls.append([h_t_entities[0], tgt])
    else:
        tgt_num = len(filtered_entities)
        for tgt_id_1 in range(tgt_num):
            for tgt_id_2 in range(tgt_id_1 + 1, tgt_num):
                tgt1 = filtered_entities[tgt_id_1]
                tgt2 = filtered_entities[tgt_id_2]
                assert tgt1 != tgt2
                _source_ls.append([tgt1, tgt2])

    neg_res = []
    for _perm in _source_ls:
        tgt_str = (h_ent_str, t_ent_str)
        tgt_ls = [
            (_perm[0], _perm[1]),
            (_perm[1], _perm[0])
        ]

        for tgt in tgt_ls:
            name_set_1 = set([_tmp_ent_mention["name"] for _tmp_ent_mention in id2ent[tgt[0]]])
            name_set_2 = set([_tmp_ent_mention["name"] for _tmp_ent_mention in id2ent[tgt[1]]])
            if h_ent_str.lower() in name_set_1 and t_ent_str.lower() in name_set_2:
                continue

            # If out of domain, ``rep_pairs`` is already empty.
            _cur_rep_pairs_copy = copy.deepcopy(rep_pairs)

            _cur_rep_pairs_copy[tgt[0]] = tgt_str[0]
            _cur_rep_pairs_copy[tgt[1]] = tgt_str[1]

            neg_res.append(_replace_entities_w_str(candidate, _cur_rep_pairs_copy))

    return neg_res


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
    h_ent_str = get_ent_str(pos_candidate, pos_candidate["h"])

    t_ent_str = get_ent_str(pos_candidate, pos_candidate["t"])

    _rep_res = replace_ent_neg_double(neg_candidate, pos_candidate["h"], h_ent_str, pos_candidate["t"], t_ent_str,
                                      rep_pairs=rep_pairs, out_of_domain=out_of_domain)

    return _rep_res


def switch_replace_neg(pos_candidate, neg_candidate, rep_pairs: Dict[int, str] = None):
    h_ent_str = get_ent_str(pos_candidate, pos_candidate["h"])

    t_ent_str = get_ent_str(pos_candidate, pos_candidate["t"])

    _rep_res = _switch_replace_neg(neg_candidate, pos_candidate["h"], h_ent_str, pos_candidate["t"], t_ent_str,
                                   rep_pairs=rep_pairs)

    return _rep_res


def get_ent_str(candi, ent_id):
    tokens = candi["sent"]
    ent_mentions = list(candi["ent"][ent_id].values())

    mention = random.choice(ent_mentions)
    return pos2str(mention["pos"][0], mention["pos"][1], tokens)


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


def rep_context_sent(selected_sentences, tgt_sent_id, rep_tgt_sent):
    context_sentences = {s_id: " ".join(s["sent"]) for s_id, s in selected_sentences.items()}
    context_sentences[tgt_sent_id] = rep_tgt_sent
    return " ".join(list(context_sentences.values()))


def context_replace_neg(context_sent, neg_candidate, rep_pairs: Dict[int, str] = None, out_of_domain: bool = False):
    _common_num = 0
    if context_sent["h"] in neg_candidate["ent"]:
        _common_num += 1
    if context_sent["t"] in neg_candidate["ent"]:
        _common_num += 1

    if _common_num == 2 and not out_of_domain:
        _rep_res = switch_replace_neg(context_sent, neg_candidate, rep_pairs=rep_pairs)
        flag = 0
    else:
        _rep_res = replace_neg(context_sent, neg_candidate, rep_pairs=rep_pairs, out_of_domain=out_of_domain)
        flag = 1

    return _rep_res, flag


def _initializer(entity_pool: Dict, negative_pool: Dict, all_neg_candidates: List, geometric_dist: torch.distributions.Distribution):
    global _entity_pool
    global _negative_pool
    global _all_neg_candidates
    global _geometric_dist

    _entity_pool = entity_pool
    _negative_pool = negative_pool
    _all_neg_candidates = all_neg_candidates
    _geometric_dist = geometric_dist


def _process_single_item(item, max_neg_num: int, aug_num: int, min_rep_num: int, shuffle_context: bool):
    examples = []
    context_examples = []

    selected_sentences = item["selected_sentences"]
    if len(selected_sentences) == 0:
        return []
    context = " ".join([" ".join(s["sent"]) for s_id, s in selected_sentences.items()])

    path_sent2rep = [(s_id, s) for s_id, s in selected_sentences.items() if s["h"] != -1 and s["t"] != -1]

    neg_candidates = [x for x in item["rest_sentences"].values() if len(x["ent"]) > 1]

    for pos_idx, pos_candi in enumerate(item["pos"]):
        # Statistics
        _res_aug = 0
        _pos_aug = 0
        _sim_aug = 0

        neg_res = []

        # Other positive candidates
        mutual_samples = [candi for candi_idx, candi in enumerate(item["pos"]) if candi_idx != pos_idx]
        for neg in mutual_samples:
            _rep_res = switch_replace_neg(pos_candi, neg, rep_pairs=None)
            if len(_rep_res) > 0:
                neg_res.extend(_rep_res)
        _pos_aug += len(neg_res)

        # Easy samples.
        for neg in neg_candidates:
            _rep_res = replace_neg(pos_candi, neg, rep_pairs=None)
            if _rep_res:
                neg_res.extend(_rep_res)
            if len(neg_res) >= max_neg_num:
                break
        _res_aug += max(len(neg_res) - _pos_aug, 0)

        if len(neg_res) > max_neg_num:
            neg_res = neg_res[:max_neg_num]

        while len(neg_res) < max_neg_num:
            neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
            while neg_data_item_id == item["id"]:
                neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
            neg = random.choice(_all_neg_candidates[neg_data_item_id])

            _rep_res = replace_neg(pos_candi, neg, rep_pairs=None, out_of_domain=True)
            if _rep_res:
                neg_res.extend(_rep_res)

        if len(neg_res) > max_neg_num:
            neg_res = neg_res[:max_neg_num]

        assert len(neg_res) == max_neg_num

        _sim_aug = max(len(neg_res) - _res_aug - _pos_aug, 0)

        examples.append({
            "context": context,
            "negative": neg_res,
            "positive": " ".join(pos_candi["sent"]),
            "orig_id": item["id"],
            "pos_aug_num": _pos_aug,
            "res_aug_num": _res_aug,
            "sim_aug_num": _sim_aug
        })

        # ============= context replaced-based examples ==================== #
        if len(path_sent2rep) == 0:
            continue

        tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
        neg_ctx_sent = []
        _ctx_pos_aug = 0
        _ctx_res_aug = 0
        _ctx_sim_aug = 0

        for neg in mutual_samples + neg_candidates:
            _rep_res, _rep_flag = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=False)

            if _rep_res:
                _left_num = min(max_neg_num, len(_rep_res))
                neg_ctx_sent.extend(_rep_res[:_left_num])
                if _rep_flag == 0:
                    _ctx_pos_aug += _left_num
                else:
                    _ctx_res_aug += _left_num

            if len(neg_ctx_sent) >= max_neg_num:
                neg_ctx_sent = neg_ctx_sent[:max_neg_num]
                break

        while len(neg_ctx_sent) < max_neg_num:
            neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
            while neg_data_item_id == item["id"]:
                neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
            neg = random.choice(_all_neg_candidates[neg_data_item_id])

            # _rep_res = replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=True)
            _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=True)
            if _rep_res:
                _left_num = max_neg_num - len(neg_ctx_sent)
                neg_ctx_sent.extend(_rep_res[:_left_num])
                _ctx_sim_aug += min(_left_num, len(_rep_res))

        context_examples.append({
            "context": context,
            "condition": " ".join(pos_candi["sent"]),
            "negative_context": [rep_context_sent(selected_sentences, tgt_ctx_sent_id, _neg_sent) for _neg_sent in neg_ctx_sent],
            "orig_id": item["id"],
            "pos_aug_num": _ctx_pos_aug,
            "res_aug_num": _ctx_res_aug,
            "sim_aug_num": _ctx_sim_aug
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

    _h_str = get_ent_str(item["pos"][0], h_t_ent_ids[0])
    _t_str = get_ent_str(item["pos"][0], h_t_ent_ids[1])

    for _ in range(aug_num):  # Repeat for augmentation

        for pos_idx, pos_candi in enumerate(item["pos"]):
            # Sample the amount of entities to be replaced from the geometric distribution.
            if min_rep_num >= len(path_ent_ids) + len(h_t_ent_ids):
                _sampled_ent_num = len(path_ent_ids) + len(h_t_ent_ids)
            else:
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

            # ======================================================================== #

            _sampled_neg_item_key = random.choice(list(_negative_pool.keys()))
            while _sampled_neg_item_key == item["id"] or _sampled_neg_item_key not in _all_neg_candidates:
                _sampled_neg_item_key = random.choice(list(_negative_pool.keys()))

            _cur_aug_rep_pairs = copy.deepcopy(sampled_rep_pairs)
            sampled_neg_candidates = _negative_pool[_sampled_neg_item_key]

            # Replace the replacement with the head/tail entity string from the sampled negative data item.
            for _tmp in h_t_ent_ids:
                if _tmp == h_t_ent_ids[0]:  # head entity
                    _neg_head_str = get_ent_str(sampled_neg_candidates[0], sampled_neg_candidates[0]["h"])
                    _cur_aug_rep_pairs[_tmp] = _neg_head_str  # If the head entity isn't to be replaced, add it.

                if _tmp == h_t_ent_ids[1]:  # tail entity
                    _neg_tail_str = get_ent_str(sampled_neg_candidates[0], sampled_neg_candidates[0]["t"])
                    _cur_aug_rep_pairs[_tmp] = _neg_tail_str  # If the head entity isn't to be replaced, add it.

            # TODO: Should other entities in ``sampled_rep_paris`` be replaced with the entity strings from the same negative sample item?

            new_sentences = []
            for _, sent in selected_sentences.items():
                new_sentences.append(_replace_entities_w_str(sent, _cur_aug_rep_pairs))

            new_pos_candi_sent = _replace_entities_w_str(pos_candi, _cur_aug_rep_pairs)

            # Statistics
            _res_aug = 0
            _pos_aug = 0
            _sim_aug = 0

            neg_res = []

            # Other positive candidates
            mutual_samples = [candi for candi_idx, candi in enumerate(item["pos"]) if candi_idx != pos_idx]
            for neg in mutual_samples:
                _rep_res = switch_replace_neg(pos_candi, neg, rep_pairs=None)
                if len(_rep_res) > 0:
                    neg_res.extend(_rep_res)
            _pos_aug += len(neg_res)

            # Add negative samples from the initial ``rest_sentences``.
            for neg in neg_candidates:
                _rep_res = replace_neg(pos_candi, neg, rep_pairs=_cur_aug_rep_pairs)
                if _rep_res:
                    neg_res.extend(_rep_res)
                if len(neg_res) >= max_neg_num:
                    break
            _res_aug += max(len(neg_res) - _pos_aug, 0)

            if len(neg_res) > max_neg_num:
                neg_res = neg_res[:max_neg_num]

            # Add simple negative samples from the positive samples of the sampled data item.
            if len(neg_res) < max_neg_num:
                for neg in sampled_neg_candidates:
                    # Add the sampled negative candidate since it contains the replaced head/tail entity already.
                    _rep_res = " ".join(neg["sent"])
                    neg_res.append(_rep_res)
                    if len(neg_res) >= max_neg_num:
                        break

            # Add simple negative samples from the ``rest_sentences`` from the sampled data item.
            if len(neg_res) < max_neg_num:
                for neg in _all_neg_candidates[_sampled_neg_item_key]:
                    _rep_res = replace_neg(sampled_neg_candidates[0], neg, rep_pairs=None, out_of_domain=False)
                    if _rep_res:
                        neg_res.extend(_rep_res)
                    if len(neg_res) >= max_neg_num:
                        break

            if len(neg_res) > max_neg_num:
                neg_res = neg_res[:max_neg_num]

            # Add simple negative samples for padding
            while len(neg_res) < max_neg_num:
                neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
                while neg_data_item_id == item["id"] or neg_data_item_id == _sampled_neg_item_key:
                    neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
                neg = random.choice(_all_neg_candidates[neg_data_item_id])

                _rep_res = replace_neg(sampled_neg_candidates[0], neg, rep_pairs=None, out_of_domain=True)
                if _rep_res:
                    neg_res.extend(_rep_res)

            if len(neg_res) > max_neg_num:
                neg_res = neg_res[:max_neg_num]

            _sim_aug += max(len(neg_res) - _pos_aug - _res_aug, 0)

            if shuffle_context:
                random.shuffle(new_sentences)
            new_context = " ".join(new_sentences)

            examples.append({
                "context": new_context,
                "negative": neg_res,
                "positive": new_pos_candi_sent,
                "orig_id": item["id"],
                "h": _cur_aug_rep_pairs[h_t_ent_ids[0]] if h_t_ent_ids[0] in _cur_aug_rep_pairs else _h_str,
                "t": _cur_aug_rep_pairs[h_t_ent_ids[1]] if h_t_ent_ids[1] in _cur_aug_rep_pairs else _t_str,
                "pos_aug_num": _pos_aug,
                "res_aug_num": _res_aug,
                "sim_aug_num": _sim_aug,
                "rep_ent_num": _sampled_ent_num
            })

            # ============= context replaced-based examples ==================== #
            # TODO: Check这里的``rep_pairs``参数是否有问题
            if len(path_sent2rep) == 0:
                continue

            tgt_ctx_sent_id, tgt_ctx_sent = random.choice(path_sent2rep)
            neg_ctx_sent = []
            _ctx_pos_aug = 0
            _ctx_res_aug = 0
            _ctx_sim_aug = 0

            for neg in mutual_samples + neg_candidates:
                _rep_res, _rep_flag = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=_cur_aug_rep_pairs, out_of_domain=False)

                if _rep_res:
                    _left_num = min(max_neg_num, len(_rep_res))
                    neg_ctx_sent.extend(_rep_res[:_left_num])
                    if _rep_flag == 0:
                        _ctx_pos_aug += _left_num
                    else:
                        _ctx_res_aug += _left_num

                if len(neg_ctx_sent) >= max_neg_num:
                    neg_ctx_sent = neg_ctx_sent[:max_neg_num]
                    break

            if len(neg_ctx_sent) < max_neg_num:
                for neg in sampled_neg_candidates + _all_neg_candidates[_sampled_neg_item_key]:
                    _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=True)

                    if _rep_res:
                        _left_num = min(max_neg_num, len(_rep_res))
                        neg_ctx_sent.extend(_rep_res[:_left_num])
                        _ctx_sim_aug += _left_num

                    if len(neg_ctx_sent) >= max_neg_num:
                        neg_ctx_sent = neg_ctx_sent[:max_neg_num]
                        break

            while len(neg_ctx_sent) < max_neg_num:
                neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
                while neg_data_item_id == item["id"]:
                    neg_data_item_id = random.choice(list(_all_neg_candidates.keys()))
                neg = random.choice(_all_neg_candidates[neg_data_item_id])

                _rep_res, _ = context_replace_neg(tgt_ctx_sent, neg, rep_pairs=None, out_of_domain=True)
                if _rep_res:
                    _left_num = max_neg_num - len(neg_ctx_sent)
                    neg_ctx_sent.extend(_rep_res[:_left_num])
                    _ctx_sim_aug += min(_left_num, len(_rep_res))

            assert len(neg_ctx_sent) == max_neg_num

            # To avoid shuffling, re-generate the new sentences
            new_sentences = dict()
            for s_id, sent in selected_sentences.items():
                new_sentences[s_id] = _replace_entities_w_str(sent, _cur_aug_rep_pairs)

            negative_context = []
            for _neg_sent in neg_ctx_sent:
                _new_context = copy.deepcopy(new_sentences)
                _new_context[tgt_ctx_sent_id] = _neg_sent
                _new_context_sentences = list(_new_context.values())
                if shuffle_context:
                    random.shuffle(_new_context_sentences)
                negative_context.append(" ".join(_new_context_sentences))

            new_sentences = list(new_sentences.values())
            if shuffle_context:
                random.shuffle(new_sentences)
            new_context = " ".join(new_sentences)

            context_examples.append({
                "context": new_context,
                "condition": new_pos_candi_sent,
                "negative_context": negative_context,
                "orig_id": item["id"],
                "pos_aug_num": _ctx_pos_aug,
                "res_aug_num": _ctx_res_aug,
                "sim_aug_num": _ctx_sim_aug
            })

    return examples, context_examples


def read_examples(file_path: str, shuffle_context: bool = False,
                  max_neg_num: int = 3, aug_num: int = 10,
                  geo_p: float = 0.5, min_rep_num: int = 1, num_workers: int = 48):
    logger.info(f"Loading raw examples from {file_path}...")
    # raw_data = json.load(open(file_path, 'r'))
    raw_data = pickle.load(open(file_path, "rb"))
    data = raw_data["examples"]
    raw_texts = raw_data["raw_texts"]

    geometric_dist = Geometric(torch.tensor([geo_p]))

    all_neg_candidates = {}
    negative_pool = {}
    _neg_cnt_1 = 0
    _neg_cnt_2 = 0
    _neg_cnt_3 = 0
    _neg_candi_cnt_2 = 0
    _neg_candi_cnt_3 = 0
    _enough = 0
    for x in tqdm(data, desc="preparing negative samples and candidates", total=len(data)):
        if x["id"] in all_neg_candidates:
            continue
        if x["id"] in negative_pool:
            continue
        # if len(x["rest_sentences"]) == 0:
        #     continue
        tmp = [y for y in x["rest_sentences"].values() if len(y["ent"]) > 1]
        if len(tmp) == 1:
            _neg_cnt_1 += 1
        elif len(tmp) == 2:
            _neg_cnt_2 += 1
        elif len(tmp) >= 3:
            _neg_cnt_3 += 1

        if len(tmp) > 0:
            all_neg_candidates[x["id"]] = tmp
            negative_pool[x["id"]] = x["pos"]
            if len(x["pos"]) == 2:
                _neg_candi_cnt_2 += 1
            elif len(x["pos"]) == 3:
                _neg_candi_cnt_3 += 1

        if len(tmp) + len(x["pos"]) >= max_neg_num + 1:
            _enough += 1
        # all_neg_candidates.extend([(x["id"], y) for y in x["rest_sentences"].values() if len(y["ent"]) > 1])
    logger.info(f"All negative candidates with size ``1``: {_neg_cnt_1}, size ``2``: {_neg_cnt_2} and ``3``: {_neg_cnt_3}")
    logger.info(f"Negative pools with size ``2``: {_neg_candi_cnt_2}, and size ``3``: {_neg_candi_cnt_3}.")
    logger.info(f"Enough negative samples: {_enough} / {len(data)} = {_enough * 1.0 / len(data)}")

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
    context_examples = []
    with Pool(num_workers, initializer=_initializer, initargs=(entity_pool, negative_pool, all_neg_candidates, geometric_dist)) as p:
        _annotate = partial(_process_single_item,
                            max_neg_num=max_neg_num, aug_num=aug_num, min_rep_num=min_rep_num, shuffle_context=shuffle_context)
        _results = list(tqdm(
            p.imap(_annotate, data, chunksize=32),
            total=len(data),
            desc="Reading examples"
        ))

    for _res, _context_res in _results:
        if _res:
            examples.extend(_res)
        if _context_res:
            context_examples.extend(_context_res)

    logger.info(f"{len(examples)} examples are loaded from {file_path}.")
    logger.info(f"{len(context_examples)} context examples are loaded from {file_path}.")

    _pos_aug = 0
    _res_aug = 0
    _sim_aug = 0
    _rep_ent_num = 0
    _rep_aug_num = 0
    for e in examples:
        _pos_aug += e.pop("pos_aug_num")
        _res_aug += e.pop("res_aug_num")
        _sim_aug += e.pop("sim_aug_num")
        if "rep_ent_num" in e:
            _rep_ent_num += e.pop("rep_ent_num")
            _rep_aug_num += 1

    _ctx_pos_aug = 0
    _ctx_res_aug = 0
    _ctx_sim_aug = 0
    for e in context_examples:
        _ctx_pos_aug += e.pop("pos_aug_num")
        _ctx_res_aug += e.pop("res_aug_num")
        _ctx_sim_aug += e.pop("sim_aug_num")

    logger.info(f"Augmentation statistics: ")
    logger.info(f"Augmentation from positive candidates: {_pos_aug} || {_pos_aug * 1.0 / len(examples)}")
    logger.info(f"Augmentation from rest sentences: {_res_aug} || {_res_aug * 1.0 / len(examples)}")
    logger.info(f"Augmentation from simple sentences: {_sim_aug} || {_sim_aug * 1.0 / len(examples)}")
    logger.info(f"Averaged replaced entity num: {_rep_ent_num} / {_rep_aug_num} || {_rep_ent_num * 1.0 / _rep_aug_num}")

    logger.info(f"Context examples statistics: ")
    logger.info(f"Augmentation from positive candidates: {_ctx_pos_aug} || {_ctx_pos_aug * 1.0 / len(context_examples)}")
    logger.info(f"Augmentation from rest sentences: {_ctx_res_aug} || {_ctx_res_aug * 1.0 / len(context_examples)}")
    logger.info(f"Augmentation from simple sentences: {_ctx_sim_aug} || {_ctx_sim_aug * 1.0 / len(context_examples)}")

    return examples, context_examples, raw_texts


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer,
                                   shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
                                   max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
                                   num_workers=48):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
                  f"{max_seq_length}_{geo_p}_{min_rep_num}_path_v8_0"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        all_examples, raw_texts = torch.load(cached_file_path)
        dataset = WikiPathDatasetV5(all_examples, raw_texts)
        return all_examples, None, dataset

    examples, context_examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num,
                                                          aug_num=aug_num, geo_p=geo_p,
                                                          min_rep_num=min_rep_num, num_workers=num_workers)
    all_examples = examples + context_examples

    # Save
    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((all_examples, raw_texts), cached_file_path)

    return all_examples, None, WikiPathDatasetV5(all_examples, raw_texts)


class WikiPathDatasetCollatorWithContext(WikiPathDatasetCollator):
    def __call__(self, batch):
        # examples, texts = list(zip(*batch))
        op_examples, ctx_examples, texts = [], [], []
        for b in batch:
            example = b.pop("example")
            if "negative_context" in example:
                ctx_examples.append(example)
            else:
                op_examples.append(example)
            # examples.append(b.pop("example"))
            texts.append(b.pop("text"))
            # assert isinstance(texts[-1], str), texts[-1]
        del batch
        batch_size = len(op_examples) + len(ctx_examples)
        assert batch_size == len(texts)

        # TODO: Check if other possible input formats are ok, e.g., <rest context> <sep> <pseudo/ground truth edge> <sep> <sep> <option>

        input_a = []
        input_b = []
        option_num = -1

        for e in op_examples:
            op = ([e["positive"]] + e["negative"])[:self.max_option_num]
            input_a.extend(op)
            input_b.extend([e["context"]] * len(op))
            if option_num == -1:
                option_num = len(op)
            else:
                assert option_num == len(op)

        for e in ctx_examples:
            positive_context = e.pop("context")
            negative_context = e.pop("negative_context")
            op = e.pop("condition")
            input_a.extend([positive_context] + negative_context)
            input_b.extend([op] * (len(negative_context) + 1))
            if option_num == -1:
                option_num = len(negative_context) + 1
            else:
                assert option_num == len(negative_context) + 1, (option_num, len(negative_context))

        option_num = min(option_num, self.max_option_num)

        tokenizer_outputs = self.tokenizer(input_a, input_b, padding=PaddingStrategy.MAX_LENGTH,
                                           truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                           return_tensors="pt")
        input_ids = tokenizer_outputs["input_ids"]
        attention_mask = tokenizer_outputs["attention_mask"]

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.MAX_LENGTH,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        res = {
            "input_ids": input_ids.reshape(batch_size, option_num, self.max_seq_length),
            "attention_mask": attention_mask.reshape(batch_size, option_num, self.max_seq_length),
            "labels": torch.zeros(batch_size, dtype=torch.long),
            "mlm_input_ids": mlm_input_ids,
            "mlm_attention_mask": mlm_attention_mask,
            "mlm_labels": mlm_labels
        }
        if "token_type_ids" in tokenizer_outputs:
            res["token_type_ids"] = tokenizer_outputs["token_type_ids"]
        return res
