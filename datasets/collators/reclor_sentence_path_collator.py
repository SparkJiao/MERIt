from typing import Dict, List, Tuple

import torch
from torch import Tensor


class ReClorSentencePathCollator:

    @staticmethod
    def process_path(_path: Tensor):
        """
        :param _path:  [batch, option_num, len]
        :return:
        """
        # [batch, option_num, len]
        _mask = (_path != -1)
        # [batch, option_num]
        _path_len = _mask.sum(dim=2)
        _b_max_path_len = _path_len.max().item()
        # Re-slice in batch
        _path = _path[:, :, :_b_max_path_len]
        _mask = _mask[:, :, :_b_max_path_len]
        _path[~_mask] = 0
        return _path, _mask, _path_len

    def __call__(self, batch: List[Tuple[Tensor, ...]]) -> Dict[str, Tensor]:
        if len(batch[0]) == 7:
            input_ids, attention_mask, token_type_ids, labels, sentence_spans, path, rev_path = list(zip(*batch))
        elif len(batch[0]) == 6:
            input_ids, attention_mask, labels, sentence_spans, path, rev_path = list(zip(*batch))
            token_type_ids = None
        else:
            raise RuntimeError()

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        sentence_spans = torch.stack(sentence_spans, dim=0)
        path = torch.stack(path, dim=0)
        rev_path = torch.stack(rev_path, dim=0)

        batch, option_num, _, _ = sentence_spans.size()
        # [batch, option_num, max_sent_num]
        max_sent_len = (sentence_spans[:, :, :, 1] - sentence_spans[:, :, :, 0]).max().item()
        # [batch, option_num, max_sent_num]
        sent_mask = (sentence_spans[:, :, :, 0] != -1)
        # [batch, option_num]
        sent_num = sent_mask.sum(dim=2)
        b_max_sent_num = sent_num.max().item()
        sentence_spans = sentence_spans[:, :, :b_max_sent_num]
        sent_mask = sent_mask[:, :, :b_max_sent_num]

        sentence_index = torch.zeros(batch, option_num, b_max_sent_num, max_sent_len, dtype=torch.long)
        sent_token_mask = torch.zeros(batch, option_num, b_max_sent_num, max_sent_len, dtype=torch.bool)
        for b_id, b_spans in enumerate(sentence_spans):
            for op_id, op_spans in enumerate(b_spans):
                for sent_id, span in enumerate(op_spans):
                    s, e = span[0].item(), span[1].item()
                    if s == -1:
                        break
                    _len = e - s
                    sentence_index[b_id, op_id, sent_id, :_len] = torch.arange(s, e, dtype=torch.long)
                    sent_token_mask[b_id, op_id, sent_id, :_len] = 1

        path, path_mask, path_len = self.process_path(path)
        rev_path, rev_path_mask, rev_path_len = self.process_path(rev_path)

        outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sentence_index": sentence_index,
            "sentence_mask": sent_mask,
            "sent_token_mask": sent_token_mask,
            "path": path,
            "path_mask": path_mask,
            "rev_path": rev_path,
            "rev_path_mask": rev_path_mask
        }
        if token_type_ids is not None:
            outputs["token_type_ids"] = torch.stack(token_type_ids, dim=0)

        return outputs
