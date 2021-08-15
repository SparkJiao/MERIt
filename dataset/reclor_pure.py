import json
import os

import torch
from typing import Tuple, Callable, Union, Dict
from torch.utils.data import TensorDataset, Dataset
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy, TensorType


class Collator:
    def __init__(self, has_token_type_id: bool = True):
        self.has_token_type_id = has_token_type_id

    def __call__(self, batch: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.has_token_type_id:
            input_ids, token_type_ids, attention_mask, labels, indices = list(zip(*batch))
            token_type_ids = torch.stack(token_type_ids, dim=0)
        else:
            input_ids, attention_mask, labels, indices = list(zip(*batch))
            token_type_ids = None

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        indices = torch.stack(indices, dim=0)

        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "indices": indices
        }
        return inputs


def get_dataset(tokenizer: PreTrainedTokenizer, file_path: str, max_seq_length: int,
                base_model_type: str) -> Tuple[Dataset, Union[None, Callable]]:
    """
    Return the ``Dataset`` and corresponding ``collate_fn``.
    :param tokenizer:
    :param file_path:
    :param max_seq_length:
    :param base_model_type:
    :return:
    """
    cached_file = f'{file_path}.{base_model_type}.{max_seq_length}'
    if os.path.exists(cached_file):
        tensors = torch.load(cached_file)
        collator = Collator(len(tensors) == 5)
        return TensorDataset(*tensors), collator

    data = json.load(open(file_path, 'r'))

    all_context = []
    all_question_option_pairs = []
    all_labels = []
    for sample in data:
        _context = sample["context"]
        _question = sample["question"]
        for option in sample["answers"]:
            all_context.append(_context)
            all_question_option_pairs.append(f'{_question}{tokenizer.sep_token}{option}')
        assert len(sample["answers"]) == 4
        if "label" not in sample:
            all_labels.append(-1)
        else:
            all_labels.append(sample["label"])

    tokenizer_outputs = tokenizer(all_question_option_pairs, all_context,
                                  padding=PaddingStrategy.MAX_LENGTH,
                                  max_length=max_seq_length,
                                  truncation=TruncationStrategy.LONGEST_FIRST,
                                  return_tensors=TensorType.PYTORCH)

    data_num = len(all_labels)

    input_ids = tokenizer_outputs["input_ids"].reshape(data_num, 4, max_seq_length)
    inputs = (input_ids,)

    if "token_type_ids" in tokenizer_outputs:
        token_type_ids = tokenizer_outputs["token_type_ids"].reshape(data_num, 4, max_seq_length)
        inputs += (token_type_ids,)

    attention_mask = tokenizer_outputs["attention_mask"].reshape(data_num, 4, max_seq_length)
    labels = torch.tensor(all_labels, dtype=torch.long)
    indices = torch.arange(data_num, dtype=torch.long)
    inputs += (attention_mask, labels, indices)

    torch.save(inputs, cached_file)

    dataset = TensorDataset(*inputs)

    collator = Collator("token_type_ids" in tokenizer_outputs)

    return dataset, collator
