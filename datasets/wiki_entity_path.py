import json
import os
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from general_util.logger import get_child_logger

logger = get_child_logger("Wiki.Entity.Path")

# _tokenizer: PreTrainedTokenizer
num_added_tokens: int = 0
ENT_1 = '[ENT_1]'
ENT_2 = '[ENT_2]'


# def _initializer(tokenizer: PreTrainedTokenizer):
#     global _tokenizer
#     _tokenizer = tokenizer


def read_examples(file_path: str):
    data = json.load(open(file_path, 'r'))

    examples = []
    for item in tqdm(data, desc='Reading examples', total=len(data)):
        examples.append({
            "context": " ".join([" ".join(sent) for sent in item["sentences"]]),
            "pos": " ".join(item["pos"]),
            "neg": " ".join(item["neg"]) if item["neg"] is not None else None
        })

    logger.info(f"{len(examples)} are loaded from {file_path}.")

    return examples


def convert_examples_into_features(file_path: str, tokenizer: PreTrainedTokenizer, max_neg_num: int = 3, max_seq_length: int = 512):
    tokenizer_name = tokenizer.__class__.__name__
    tokenizer_name = tokenizer_name.replace('TokenizerFast', '')
    tokenizer_name = tokenizer_name.replace('Tokenizer', '').lower()

    global num_added_tokens
    num_added_tokens += tokenizer.add_tokens([ENT_1, ENT_2])

    logger.info(f"Extend vocabulary with [{num_added_tokens}].")

    file_suffix = f"{tokenizer_name}_{max_neg_num}_{max_seq_length}_path"
    cached_file_path = f"{file_path}_{file_suffix}"
    if os.path.exists(cached_file_path):
        logger.info(f"Loading cached file from {cached_file_path}")
        examples, features, tensors = torch.load(cached_file_path)
        return examples, features, tensors

    examples = read_examples(file_path)

    data_num = len(examples)

    sentences = [e["context"] for e in examples]
    pos = [e["pos"] for e in examples]

    neg = []
    for e in examples:
        if e["neg"] is None:
            neg_samples = random.sample(pos, max_neg_num)
        else:
            neg_samples = [e["neg"]] + random.sample(pos, max_neg_num - 1)
        neg.append(neg_samples)

    ex_sentences = []
    options = []
    for sent, pos_s, neg_group in zip(sentences, pos, neg):
        ex_sentences.extend([sent] * (max_neg_num + 1))
        options.extend([pos_s] + neg_group)

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


def get_num_extended_tokens():
    return num_added_tokens


class WikiPathDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> T_co:
        return self.examples[index]
