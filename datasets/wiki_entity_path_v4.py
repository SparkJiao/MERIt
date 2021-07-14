import json
import os
import random
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from general_util.logger import get_child_logger

logger = get_child_logger("Wiki.Entity.Path.V4")


def replace_ent(candidate, h_ent_id, h_ent_str, t_ent_id, t_ent_str):
    tokens = candidate["sent"]
    entities = candidate["ent"]

    # logger.info(entities)

    filtered_entities = [ent for ent_id, ent in entities if ent_id not in [h_ent_id, t_ent_id]]
    h_t_entities = [ent for ent_id, ent in entities if ent_id in [h_ent_id, t_ent_id]]

    if len(filtered_entities) == 0:
        return None
    elif len(filtered_entities) == 1:
        if len(h_t_entities) == 0:
            return None
        else:
            filtered_entities.append(random.choice(h_t_entities))

    re1, re2 = random.sample(filtered_entities, 2)
    re = sorted([re1, re2], key=lambda x: x["pos"][0])

    tgt = [h_ent_str, t_ent_str]
    # random.shuffle(tgt)
    # assert not (h_ent_str.lower() == re[0]["name"] and t_ent_str.lower() == re[1]["name"]), (h_ent_str.lower(), t_ent_str.lower(), re)
    if h_ent_str.lower() == re[0]["name"] and t_ent_str.lower() == re[1]["name"]:
        return None

    new_tokens = []
    _last_e = 0
    for r in re:
        s, e = r["pos"]
        new_tokens.extend(tokens[_last_e: s])
        new_tokens.append(tgt.pop())
        _last_e = e

    new_tokens.extend(tokens[_last_e:])
    return " ".join(new_tokens)


def _replace_entity_w_str(tokens, entities, entity_str):
    entities = sorted(list(zip(entities, entity_str)), key=lambda x: x[0]["pos"][0])

    new_tokens = []
    _last_e = 0
    for ent, ent_str in entities:
        s, e = ent["pos"]
        new_tokens.extend(tokens[_last_e: s])
        new_tokens.append(ent_str)
        _last_e = e

    new_tokens.extend(tokens[_last_e:])
    return " ".join(new_tokens)


def replace_neg(pos_candidate, neg_candidate):
    h_ent_id, h_ent = random.choice(pos_candidate["h"])
    h_ent_str = pos2str(h_ent["pos"][0], h_ent["pos"][1], pos_candidate["sent"])
    t_ent_id, t_ent = random.choice(pos_candidate["t"])
    t_ent_str = pos2str(t_ent["pos"][0], t_ent["pos"][1], pos_candidate["sent"])
    _rep_res = replace_ent(neg_candidate, h_ent_id, h_ent_str, t_ent_id, t_ent_str)
    return _rep_res


def pos2str(ent_s, ent_e, tokens):
    return " ".join(tokens[ent_s: ent_e])


def sample_entity(pool, src_id, k):
    # range_ls = list(range(len(pool)))
    # range_ls = range_ls[:src_id] + range_ls[(src_id + 1):]
    # entity_ls = pool[random.choice(range_ls)]
    # logger.info(entity_ls)
    # return random.sample(entity_ls, k)
    return random.sample(pool, k)


def read_examples(file_path: str, max_neg_num: int = 3, aug_num: int = 10):
    data = json.load(open(file_path, 'r'))

    all_neg_candidates = []
    for x in data:
        all_neg_candidates.extend([y for y in x["rest_sentences"].values() if len(y["ent"]) > 1])

    # entity_pool = {}
    entity_pool = set()
    for data_id, x in enumerate(data):
        # pos_candi = x["pos"][0]
        tmp_ls = []
        for pos_candi in x["pos"]:
            for h_ent_id, h_ent in pos_candi["h"]:
                tmp_ls.append(pos2str(h_ent["pos"][0], h_ent["pos"][1], pos_candi["sent"]))
            for t_ent_id, t_ent in pos_candi["t"]:
                tmp_ls.append(pos2str(t_ent["pos"][0], t_ent["pos"][1], pos_candi["sent"]))
        # entity_pool[data_id] = list(set(tmp_ls))
        entity_pool.update(set(tmp_ls))
    entity_pool = list(entity_pool)

    examples = []
    for item_id, item in enumerate(tqdm(data, desc='Reading examples', total=len(data))):
        selected_sentences = item["selected_sentences"]
        context = " ".join([" ".join(s["sent"]) for s_id, s in selected_sentences.items()])

        neg_candidates = [x for x in item["rest_sentences"].values() if len(x["ent"]) > 1]

        for pos_idx, pos_candi in enumerate(item["pos"]):

            neg_res = []
            for neg in neg_candidates:
                _rep_res = replace_neg(pos_candi, neg)
                if _rep_res is not None:
                    neg_res.append(_rep_res)
                    if len(neg_res) == max_neg_num:
                        break

            while len(neg_res) < max_neg_num:
                neg = random.choice(all_neg_candidates)
                _rep_res = replace_neg(pos_candi, neg)
                if _rep_res is not None:
                    neg_res.append(_rep_res)
                    if len(neg_res) == max_neg_num:
                        break

            examples.append({
                "context": context,
                "negative": neg_res,
                "positive": " ".join(pos_candi["sent"])
            })

        # Augment the context
        # 1. Choose the head entity or the tail entity as the target entity.
        # 2. Random sample another entity from the entity pool.
        # 3. Replace the target entity in the context with the sampled entity.
        # 4. Replace the target entity in negative samples with the sampled entity.
        h_ent_id = item["pos"][0]["h"][0][0]
        t_ent_id = item["pos"][0]["t"][0][0]
        his_tgt_entity = set()
        for _ in range(aug_num):  # Repeat for augmentation
            for src_id, src_ent_id in enumerate([h_ent_id, t_ent_id]):
                tgt_entity: str = sample_entity(entity_pool, item_id, 1)[0]
                while tgt_entity in his_tgt_entity:
                    tgt_entity: str = sample_entity(entity_pool, item_id, 1)[0]
                his_tgt_entity.add(tgt_entity)

                new_sentences = []
                for _, s in selected_sentences.items():
                    # Find the matched head entity or tail entity.
                    src_ent_ls = [s_ent for s_ent_id, s_ent in s["ent"] if s_ent_id == src_ent_id]
                    new_sentences.append(_replace_entity_w_str(s["sent"], src_ent_ls, [tgt_entity] * len(src_ent_ls)))
                new_context = " ".join(new_sentences)

                for pos_idx, pos_candi in enumerate(item["pos"]):

                    _, h_ent = random.choice(pos_candi["h"])
                    h_ent_str = pos2str(h_ent["pos"][0], h_ent["pos"][1], pos_candi["sent"])
                    _, t_ent = random.choice(pos_candi["t"])
                    t_ent_str = pos2str(t_ent["pos"][0], t_ent["pos"][1], pos_candi["sent"])

                    ini_h_t_ls = [(h_ent_id, h_ent_str), (t_ent_id, t_ent_str)]
                    # logger.info(ini_h_t_ls)
                    ini_h_t_ls[src_id] = (-1, tgt_entity)
                    # logger.info(ini_h_t_ls)

                    # logger.info(tgt_entity)
                    neg_res = []

                    # Generate a negative sample from the positive sample
                    target_ls = pos_candi["h"] if src_id == 0 else pos_candi["t"]
                    source_ls = pos_candi["t"] if src_id == 0 else pos_candi["h"]
                    # rep_ent_str = [pos2str(x["pos"][0], x["pos"][1], pos_candi["sent"]) for idx, x in contra_ls]
                    rep_ent_str = [pos2str(source_ls[0][1]["pos"][0], source_ls[0][1]["pos"][1], pos_candi["sent"])] * len(target_ls)
                    pos_src_ent_ls_false = [x for idx, x in target_ls + source_ls]
                    rep_ent_str_ls_false = rep_ent_str + [tgt_entity] * len(source_ls)
                    aug_neg = _replace_entity_w_str(pos_candi["sent"], pos_src_ent_ls_false, rep_ent_str_ls_false)
                    neg_res.append(aug_neg)

                    for neg in neg_candidates:
                        _rep_res = replace_ent(neg, ini_h_t_ls[0][0], ini_h_t_ls[0][1], ini_h_t_ls[1][0], ini_h_t_ls[1][1])
                        if _rep_res is not None:
                            neg_res.append(_rep_res)
                            if len(neg_res) == max_neg_num:
                                break

                    while len(neg_res) < max_neg_num:
                        neg = random.choice(all_neg_candidates)
                        _rep_res = replace_ent(neg, ini_h_t_ls[0][0], ini_h_t_ls[0][1], ini_h_t_ls[1][0], ini_h_t_ls[1][1])
                        if _rep_res is not None:
                            neg_res.append(_rep_res)
                            if len(neg_res) == max_neg_num:
                                break

                    pos_src_ent_ls = pos_candi["h"] if src_id == 0 else pos_candi["t"]
                    pos_src_ent_ls = [x for idx, x in pos_src_ent_ls]
                    positive = _replace_entity_w_str(pos_candi["sent"], pos_src_ent_ls, [tgt_entity] * len(pos_src_ent_ls))

                    examples.append({
                        "context": new_context,
                        "negative": neg_res,
                        "positive": positive
                    })

    logger.info(f"{len(examples)} are loaded from {file_path}.")

    return examples


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, max_neg_num: int = 3, aug_num: int = 10,
                                   max_seq_length: int = 512):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{max_neg_num}_{aug_num}_{max_seq_length}_path"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        examples, features, tensors = torch.load(cached_file_path)
        return examples, features, tensors

    examples = read_examples(file_path, max_neg_num=max_neg_num, aug_num=aug_num)

    data_num = len(examples)

    ex_sentences = []
    options = []
    for example in examples:
        ex_sentences.extend([example["context"]] * (max_neg_num + 1))
        options.extend([example["positive"]] + example["negative"])
    assert len(ex_sentences) == len(options), (len(ex_sentences), len(options))

    logger.info("Tokenization...")

    tokenizer_outputs = tokenizer(ex_sentences, text_pair=options, truncation=TruncationStrategy.ONLY_FIRST, return_tensors="pt",
                                  max_length=max_seq_length, padding=PaddingStrategy.MAX_LENGTH)

    input_ids = tokenizer_outputs["input_ids"].reshape(data_num, max_neg_num + 1, -1)
    attention_mask = tokenizer_outputs["attention_mask"].reshape(data_num, max_neg_num + 1, -1)
    tensors = (input_ids, attention_mask,)

    if "token_type_ids" in tokenizer_outputs:
        token_type_ids = tokenizer_outputs["token_type_ids"].reshape(data_num, max_neg_num + 1, -1)
        tensors = tensors + (token_type_ids,)

    labels = torch.zeros(data_num, dtype=torch.long)
    tensors = tensors + (labels,)

    logger.info(f"Size of ``input_ids``: {input_ids.size()}.")

    # Save
    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((examples, None, tensors), cached_file_path)

    return examples, None, tensors
