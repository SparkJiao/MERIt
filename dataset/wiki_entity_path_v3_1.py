import json
import os
import random

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from general_util.logger import get_child_logger

logger = get_child_logger("Wiki.Entity.Path.V3.1")


def replace_ent_neg(candidate, h_ent, t_ent):
    tokens = candidate["sent"]
    entities = candidate["ent"]

    re1, re2 = random.sample(entities, 2)
    re = sorted([re1, re2], key=lambda x: x["pos"][0])

    # _seed = random.random()
    # if _seed < 0.5:
    #     h_ent, t_ent = t_ent, h_ent
    tgt = [h_ent, t_ent]
    random.shuffle(tgt)
    if h_ent.lower() == re[0]["name"] and t_ent.lower() == re[1]["name"]:
        random.shuffle(tgt)
    # assert not (h_ent.lower() == re[0]["name"] and t_ent.lower() == re[1]["name"]), (h_ent.lower(), t_ent.lower(), re)

    new_tokens = []
    _last_e = 0
    for r in re:
        s, e = r["pos"]
        new_tokens.extend(tokens[_last_e: s])
        new_tokens.append(tgt.pop())
        _last_e = e

    new_tokens.extend(tokens[_last_e:])
    return " ".join(new_tokens)


def replace_ent_pos(candidate, re_0, re_1, h_ent, t_ent):
    tokens = candidate["sent"]
    # entities = candidate["ent"]

    # re1, re2 = random.sample(entities, 2)
    # re = sorted([re1, re2], key=lambda x: x["pos"][0])
    re = sorted([re_0, re_1], key=lambda x: x["pos"][0])

    # _seed = random.random()
    # if _seed < 0.5:
    #     h_ent, t_ent = t_ent, h_ent
    tgt = [h_ent, t_ent]
    # random.shuffle(tgt)
    # if h_ent.lower() == re[0]["name"] and t_ent.lower() == re[1]["name"]:
    #     random.shuffle(tgt)
    assert not (h_ent.lower() == re[0]["name"] and t_ent.lower() == re[1]["name"]), (h_ent.lower(), t_ent.lower(), re)

    new_tokens = []
    _last_e = 0
    for r in re:
        s, e = r["pos"]
        new_tokens.extend(tokens[_last_e: s])
        new_tokens.append(tgt.pop())
        _last_e = e

    new_tokens.extend(tokens[_last_e:])
    return " ".join(new_tokens)


def read_examples(file_path: str, max_neg_num: int = 3):
    data = json.load(open(file_path, 'r'))

    all_neg_candidates = []
    for x in data:
        all_neg_candidates.extend([y for y in x["rest_sentences"].values() if len(y["ent"]) > 1])

    examples = []
    for item in tqdm(data, desc='Reading examples', total=len(data)):
        selected_sentences = item["selected_sentences"]
        context = " ".join([" ".join(s["sent"]) for s_id, s in selected_sentences.items()])

        neg_candidates = [x for x in item["rest_sentences"].values() if len(x["ent"]) > 1]
        if len(neg_candidates) < max_neg_num:
            tmp_neg_candi = random.sample(all_neg_candidates, max_neg_num - len(neg_candidates))
            neg_candidates += tmp_neg_candi
        neg_candidates = neg_candidates[:max_neg_num]

        # This overlooks the case of entity.
        # h_ent = item["pos"][0]["h"][0]["name"]
        # t_ent = item["pos"][0]["t"][0]["name"]
        h_ent_s, h_ent_e = item["pos"][0]["h"][0]["pos"]
        h_ent = " ".join(item["pos"][0]["sent"][h_ent_s: h_ent_e])
        h_ent_name = item["pos"][0]["h"][0]["name"]
        t_ent_s, t_ent_e = item["pos"][0]["t"][0]["pos"]
        t_ent = " ".join(item["pos"][0]["sent"][t_ent_s: t_ent_e])
        t_ent_name = item["pos"][0]["t"][0]["name"]

        # print(h_ent, h_ent_name, t_ent, t_ent_name)

        path_entities = []
        for _s_sent in selected_sentences.values():
            for ent in _s_sent["ent"]:
                ent_name = ent["name"]
                if ent_name == h_ent.lower() or ent_name == t_ent.lower():
                    continue
                if ent_name == h_ent_name or ent_name == t_ent_name:
                    continue
                ent_s, ent_e = ent["pos"]
                ent_raw = " ".join(_s_sent["sent"][ent_s: ent_e])
                if ent_raw.lower() == h_ent.lower() or ent_raw.lower() == t_ent.lower():
                    continue
                if ent_raw.lower() == h_ent_name or ent_raw.lower() == t_ent_name:
                    continue
                path_entities.append(ent_raw)

        neg_candi_res = [replace_ent_neg(neg_candi, h_ent, t_ent) for neg_candi in neg_candidates]
        # print(len(neg_candi_res))
        # assert len(neg_res) == max_neg_num, len(neg_res)
        # if len(path_entities) * (len(path_entities) - 1) < max_neg_num:
        #     continue

        for pos_idx, pos_candi in enumerate(item["pos"]):

            pos_sent = " ".join(pos_candi["sent"])

            pos_h_ent = pos_candi["h"][0]["name"]
            pos_t_ent = pos_candi["t"][0]["name"]
            pos_h_ent_raw = " ".join(pos_candi["sent"][pos_candi["h"][0]["pos"][0]: pos_candi["h"][0]["pos"][1]])
            pos_t_ent_raw = " ".join(pos_candi["sent"][pos_candi["t"][0]["pos"][0]: pos_candi["t"][0]["pos"][1]])

            neg_res = []
            for ent_idx_1, ent_1 in enumerate(path_entities):
                if ent_1.lower() in [pos_h_ent, pos_t_ent, pos_t_ent_raw.lower(), pos_h_ent_raw.lower()]:
                    continue
                for ent_idx_2, ent_2 in enumerate(path_entities):
                    if ent_2.lower() in [pos_h_ent, pos_t_ent, pos_t_ent_raw.lower(), pos_h_ent_raw.lower()]:
                        continue

                    neg_res.append(replace_ent_pos(pos_candi, pos_candi["h"][0], pos_candi["t"][0], ent_1, ent_2))

                    if len(neg_res) == max_neg_num:
                        break
                if len(neg_res) == max_neg_num:
                    break
            if len(neg_res) < max_neg_num:
                neg_res.extend(random.sample(neg_candi_res, max_neg_num - len(neg_res)))

            assert len(neg_res) == max_neg_num

            examples.append({
                "context": context,
                "negative": neg_res,
                "positive": pos_sent
            })

    logger.info(f"{len(examples)} are loaded from {file_path}.")

    return examples


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, max_neg_num: int = 3, max_seq_length: int = 512):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{max_neg_num}_{max_seq_length}_path"
    cached_file_path = f"{file_path}_{file_suffix}_v3.1"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        examples, features, tensors = torch.load(cached_file_path)
        return examples, features, tensors

    examples = read_examples(file_path, max_neg_num=max_neg_num)

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
