from typing import Tuple, List, Dict, Optional

import torch
from torch import Tensor
from transformers import AutoTokenizer, RobertaTokenizer
from torch.utils.data.dataset import Dataset, T_co, TensorDataset


class BaseDictCollator:
    def __init__(self, add_mlm_labels: bool = False, mlm_probability: float = 0.15, tokenizer: str = 'bert-base-uncased'):
        self.add_mlm_labels = add_mlm_labels
        self.mlm_probability = mlm_probability
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __call__(self, batch: List[Tuple[Tensor, ...]]) -> Dict[str, Tensor]:
        if len(batch[0]) == 4:
            input_ids, attention_mask, token_type_ids, labels = list(zip(*batch))
            mlm_input_ids = None
            mlm_attention_mask = None
        elif len(batch[0]) == 3:
            input_ids, attention_mask, labels = list(zip(*batch))
            token_type_ids = None
            mlm_input_ids = None
            mlm_attention_mask = None
        elif len(batch[0]) == 6:
            input_ids, attention_mask, token_type_ids, labels, mlm_input_ids, mlm_attention_mask = list(zip(*batch))
        elif len(batch[0]) == 5:
            input_ids, attention_mask, labels, mlm_input_ids, mlm_attention_mask = list(zip(*batch))
            token_type_ids = None
        else:
            raise RuntimeError()

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)

        outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if token_type_ids is not None:
            outputs["token_type_ids"] = torch.stack(token_type_ids, dim=0)

        if self.add_mlm_labels:
            if mlm_input_ids is None:
                mlm_input_ids = input_ids[:, 0].clone()
                mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)

                outputs["mlm_input_ids"] = mlm_input_ids
                outputs["mlm_labels"] = mlm_labels
            else:
                mlm_input_ids = torch.stack(mlm_input_ids, dim=0)
                mlm_attention_mask = torch.stack(mlm_attention_mask, dim=0)

                mlm_input_ids, mlm_labels = self.mask_tokens(mlm_input_ids)
                outputs["mlm_input_ids"] = mlm_input_ids
                outputs["mlm_attention_mask"] = mlm_attention_mask
                outputs["mlm_labels"] = mlm_labels

        return outputs

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


class UnalignedTensorDataset(Dataset):

    def __init__(self, tensor_groups: Tuple[Tuple[Tensor, ...], ...], id_for_len: int):
        self.length = tensor_groups[id_for_len][0].size(0)
        self.tensors = []
        for _tensors in tensor_groups:
            assert all(_tensors[0].size(0) == _tensor.size(0) for _tensor in _tensors), "Size mismatch between tensors"
            self.tensors.extend(_tensors)

    def __getitem__(self, index) -> T_co:
        res = []
        for _tensor in self.tensors:
            res.append(_tensor[index % _tensor.size(0)])
        return res

    def __len__(self):
        return self.length
