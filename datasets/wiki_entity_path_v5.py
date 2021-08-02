import copy
import os
import pickle
import random
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

import torch
from torch.distributions.geometric import Geometric
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from general_util.logger import get_child_logger

logger = get_child_logger("Wiki.Entity.Path.V5")
_tokenizer: PreTrainedTokenizer


# def replace_ent(candidate, h_ent_id, h_ent_str, t_ent_id, t_ent_str):
#     tokens = candidate["sent"]
#     entities = candidate["ent"]
#
#     # logger.info(entities)
#
#     filtered_entities = [ent_pos_dic for ent_id, ent_pos_dic in entities.items() if ent_id not in [h_ent_id, t_ent_id]]
#     h_t_entities = []
#     for _tmp in [h_ent_id, t_ent_id]:
#         if _tmp in entities:
#             h_t_entities.append(entities[_tmp])
#     assert len(h_t_entities) < 2
#
#     # filtered_entities = [ent for ent_id, ent in entities if ent_id not in [h_ent_id, t_ent_id]]
#     # h_t_entities = [ent for ent_id, ent in entities if ent_id in [h_ent_id, t_ent_id]]
#
#     if len(filtered_entities) == 0:
#         return None
#     # elif len(filtered_entities) == 1:
#     # if len(h_t_entities) == 0:
#     #     return None
#     # else:
#     #     filtered_entities.append(random.choice(h_t_entities))
#     else:
#         # If there is already an head/tail entity in the negative sentence, we only sample another entity to replace.
#         if len(h_t_entities) == 1:
#             re1 = h_t_entities[0]
#             re2 = random.choice(filtered_entities)
#         else:
#             re1, re2 = random.sample(filtered_entities, 2)
#
#     # re1, re2 = random.sample(filtered_entities, 2)
#     # re = sorted([re1, re2], key=lambda x: x["pos"][0])
#     # Tight the target ``id`` used for replacement for the same entity.
#     all_ent_pos = []
#     for r in re1.values():
#         r["tgt"] = 0
#         all_ent_pos.append(r)
#     for r in re2.values():
#         r["tgt"] = 1
#         all_ent_pos.append(r)
#     # all_ent_pos = list(re1.values()) + list(re2.values())
#     re = sorted(all_ent_pos, key=lambda x: x["pos"][0])
#     # Non-overlapping check.
#     for _tmp_id, _tmp in enumerate(re):
#         if _tmp_id == 0:
#             continue
#         assert _tmp["pos"][0] >= re[_tmp_id - 1]["pos"][1]
#
#     tgt = [h_ent_str, t_ent_str]
#     random.shuffle(tgt)
#     # assert not (h_ent_str.lower() == re[0]["name"] and t_ent_str.lower() == re[1]["name"]), (h_ent_str.lower(), t_ent_str.lower(), re)
#     # if h_ent_str.lower() == re[0]["name"] and t_ent_str.lower() == re[1]["name"]:
#     #     return None
#     _same_n = 0
#     for _tmp in tgt:
#         for _tmp_re in re:
#             if _tmp == _tmp_re["name"]:
#                 _same_n += 1
#                 break
#     if _same_n >= 2:
#         logger.warning(f"Same replacement found. ({tgt}, {re[0]['name']}, {re[1]['name']}) Continue.")
#         return None
#
#     new_tokens = []
#     _last_e = 0
#     for r in re:
#         s, e = r["pos"]
#         new_tokens.extend(tokens[_last_e: s])
#         new_tokens.append(tgt[r["tgt"]])
#         _last_e = e
#
#     new_tokens.extend(tokens[_last_e:])
#     return " ".join(new_tokens)


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


# def _replace_entity_w_str(tokens, entities, entity_str):
#     entities = sorted(list(zip(entities, entity_str)), key=lambda x: x[0]["pos"][0])
#
#     new_tokens = []
#     _last_e = 0
#     for ent, ent_str in entities:
#         s, e = ent["pos"]
#         new_tokens.extend(tokens[_last_e: s])
#         new_tokens.append(ent_str)
#         _last_e = e
#
#     new_tokens.extend(tokens[_last_e:])
#     return " ".join(new_tokens)


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


def read_examples(file_path: str, shuffle_context: bool = False,
                  max_neg_num: int = 3, aug_num: int = 10,
                  geo_p: float = 0.5, min_rep_num: int = 1):
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
    for item_id, item in enumerate(tqdm(data, desc='Reading examples', total=len(data))):
        selected_sentences = item["selected_sentences"]
        if len(selected_sentences) == 0:
            continue
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
                neg_data_id, neg = random.choice(all_neg_candidates)
                while neg_data_id == item["id"]:
                    neg_data_id, neg = random.choice(all_neg_candidates)
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

        for _ in range(aug_num):  # Repeat for augmentation

            # Sample the amount of entities to be replaced from the geometric distribution.
            _sampled_ent_num = int(geometric_dist.sample().item()) + min_rep_num
            while _sampled_ent_num >= len(path_ent_ids):
                _sampled_ent_num = int(geometric_dist.sample().item()) + min_rep_num
            assert min_rep_num <= _sampled_ent_num < len(path_ent_ids)

            sampled_ent_ids = random.sample(list(path_ent_ids), _sampled_ent_num)
            # TODO: How to check there is no repetition in the sampled entity strings ?
            target_ent_str = sample_entity(entity_pool, item["id"], _sampled_ent_num)
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
                    neg_data_id, neg = random.choice(all_neg_candidates)
                    while neg_data_id == item["id"]:
                        neg_data_id, neg = random.choice(all_neg_candidates)
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

    logger.info(f"{len(examples)} examples are loaded from {file_path}.")

    return examples, raw_texts


def _initializer(tokenizer: PreTrainedTokenizer):
    global _tokenizer
    _tokenizer = tokenizer


def _convert_example_into_features(example, max_seq_length: int = 512):
    context = example[0]
    option = example[1]

    tokenizer_outputs = _tokenizer(context, text_pair=option, truncation=TruncationStrategy.LONGEST_FIRST,
                                   padding=PaddingStrategy.MAX_LENGTH, max_length=max_seq_length)
    return tokenizer_outputs


def _convert_raw_text_into_mlm(raw_text: str, max_seq_length: int = 512):
    tokenizer_outputs = _tokenizer(raw_text, truncation=TruncationStrategy.ONLY_FIRST,
                                   padding=PaddingStrategy.MAX_LENGTH, max_length=max_seq_length)
    return tokenizer_outputs


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer,
                                   shuffle_context: bool = False, max_neg_num: int = 3, aug_num: int = 10,
                                   max_seq_length: int = 512, geo_p: float = 0.5, min_rep_num: int = 1,
                                   num_workers: int = 48):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    file_suffix = f"{tokenizer_name}_{shuffle_context}_{max_neg_num}_{aug_num}_" \
                  f"{max_seq_length}_{geo_p}_{min_rep_num}_path_v5_0"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        examples, features, tensors = torch.load(cached_file_path)
        return examples, features, tensors

    examples, raw_texts = read_examples(file_path, shuffle_context=shuffle_context, max_neg_num=max_neg_num, aug_num=aug_num,
                                        geo_p=geo_p, min_rep_num=min_rep_num)

    data_num = len(examples)

    ex_sentences = []
    options = []
    for example in examples:
        ex_sentences.extend([example["context"]] * (max_neg_num + 1))
        options.extend([example["positive"]] + example["negative"])
    assert len(ex_sentences) == len(options), (len(ex_sentences), len(options))
    # del examples

    with Pool(num_workers, initializer=_initializer, initargs=(tokenizer,)) as p:
        _annotate = partial(_convert_example_into_features, max_seq_length=max_seq_length)
        _results = list(tqdm(
            p.imap(_annotate, zip(ex_sentences, options), chunksize=32),
            total=len(ex_sentences),
            desc='Tokenization'
        ))
    del ex_sentences
    del options

    input_ids = torch.tensor([o["input_ids"] for o in _results], dtype=torch.long).reshape(data_num, max_neg_num + 1, -1)
    attention_mask = torch.tensor([o["attention_mask"] for o in _results], dtype=torch.long).reshape(data_num, max_neg_num + 1, -1)

    tensors = (input_ids, attention_mask,)

    if "token_type_ids" in _results[0]:
        token_type_ids = torch.tensor([o["token_type_ids"] for o in _results], dtype=torch.long)
        tensors = tensors + (token_type_ids,)

    labels = torch.zeros(data_num, dtype=torch.long)
    tensors = tensors + (labels,)

    logger.info(f"Size of ``input_ids``: {input_ids.size()}.")
    del _results

    with Pool(num_workers, initializer=_initializer, initargs=(tokenizer,)) as p:
        _annotate = partial(_convert_raw_text_into_mlm, max_seq_length=max_seq_length)
        _mlm_results = list(tqdm(
            p.imap(_annotate, raw_texts, chunksize=32),
            total=len(raw_texts),
            desc="Tokenization for MLM"
        ))
    del raw_texts

    _aligned_mlm_results = []
    while len(_aligned_mlm_results) < data_num:
        diff = data_num - len(_aligned_mlm_results)
        if diff < len(_mlm_results):
            _aligned_mlm_results.extend(random.sample(_mlm_results, diff))
        else:
            _aligned_mlm_results.extend(_mlm_results[:])
    assert len(_aligned_mlm_results) == data_num

    mlm_input_ids = torch.tensor([o["input_ids"] for o in _aligned_mlm_results], dtype=torch.long)
    mlm_attention_mask = torch.tensor([o["attention_mask"] for o in _aligned_mlm_results], dtype=torch.long)

    logger.info(f"Size of ``mlm_input_ids``: {mlm_input_ids.size()}")

    tensors = tensors + (mlm_input_ids, mlm_attention_mask)

    # Save
    logger.info(f"Saving processed features into {cached_file_path}.")
    torch.save((examples, None, tensors), cached_file_path)

    return examples, None, tensors


class WikiPathDatasetV5(Dataset):
    def __init__(self, examples, raw_texts):
        self.examples = examples

        _aligned_texts = []
        while len(_aligned_texts) < len(examples):
            diff = len(examples) - len(_aligned_texts)
            if diff < len(raw_texts):
                _aligned_texts.extend(random.sample(raw_texts, diff))
            else:
                _aligned_texts.extend(raw_texts[:])
        assert len(_aligned_texts) == len(self.examples)

        cnt = Counter(list(map(lambda x: len(x["negative"]) if "negative" in x else len(x["negative_context"]), examples)))
        assert len(cnt) == 1, cnt

        self.raw_texts = _aligned_texts

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> T_co:
        example = self.examples[index]
        text = self.raw_texts[index]
        return {
            "example": example,
            "text": text
        }


class WikiPathDatasetCollator:
    def __init__(self, max_seq_length: int, tokenizer: str, mlm_probability: float = 0.15, max_option_num: int = 4):
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mlm_probability = mlm_probability
        self.max_option_num = max_option_num

    def __call__(self, batch):
        # examples, texts = list(zip(*batch))
        examples, texts = [], []
        for b in batch:
            # assert list(b.keys()) == ["example", "text"], b.keys()
            examples.append(b.pop("example"))
            texts.append(b.pop("text"))
            # assert isinstance(texts[-1], str), texts[-1]
        del batch

        sentences = []
        options = []
        for e in examples:
            op = ([e["positive"]] + e["negative"])[:self.max_option_num]
            options.extend(op)
            sentences.extend([e["context"]] * len(op))
        batch_size = len(examples)
        # option_num = len(examples[0]["negative"]) + 1
        option_num = min(len(examples[0]["negative"]) + 1, self.max_option_num)

        tokenizer_outputs = self.tokenizer(sentences, options, padding=PaddingStrategy.MAX_LENGTH,
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

    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            # Remove padding.
            special_tokens_mask = special_tokens_mask | (labels == self.tokenizer.pad_token_id)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class WikiPathDatasetCollatorOnlyMLM(WikiPathDatasetCollator):
    def __call__(self, batch):
        texts = []
        for b in batch:
            texts.append(b.pop("text"))
        del batch

        mlm_tokenize_outputs = self.tokenizer(texts, padding=PaddingStrategy.MAX_LENGTH,
                                              truncation=TruncationStrategy.LONGEST_FIRST, max_length=self.max_seq_length,
                                              return_tensors="pt")
        mlm_input_ids = mlm_tokenize_outputs["input_ids"]
        mlm_attention_mask = mlm_tokenize_outputs["attention_mask"]

        mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

        return {
            "input_ids": mlm_input_ids,
            "attention_mask": mlm_attention_mask,
            "labels": mlm_labels
        }
