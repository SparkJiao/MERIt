from torch import Tensor
import torch
from typing import Tuple, List, Dict


class BaseDictCollator:
    def __call__(self, batch: List[Tuple[Tensor, ...]]) -> Dict[str, Tensor]:
        if len(batch[0]) == 4:
            input_ids, attention_mask, token_type_ids, labels = list(zip(*batch))
        elif len(batch[0]) == 3:
            input_ids, attention_mask, labels = list(zip(*batch))
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

        return outputs
