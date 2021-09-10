import torch
from torch import Tensor


def extract_sent_tokens(source: Tensor, sentence_index: Tensor, sent_token_mask: Tensor, sentence_ids: Tensor, sentence_ids_mask: Tensor):
    """
    :param source: [batch, seq_len]
    :param sentence_index: [batch, max_sent_num, max_sent_len]
    :param sent_token_mask: [batch, max_sent_num, max_sent_len]
    :param sentence_ids: [batch, path_len]
    :param sentence_ids_mask: [batch, path_len]
    :return:
    """
    batch = sentence_index.size(0)
    max_sent_len = sentence_index.size(-1)
    path_len = sentence_ids.size(1)
    ex_sentence_ids = sentence_ids.unsqueeze(-1).expand(-1, -1, max_sent_len)
    ex_sentence_ids_mask = sentence_ids_mask.unsqueeze(-1).expand(-1, -1, max_sent_len)
    # [batch, path_len, max_sent_len]
    gathered_sent_token_ids = torch.gather(sentence_index, dim=1, index=ex_sentence_ids).reshape(batch, -1)
    gathered_sent_token_mask = torch.gather(sent_token_mask, dim=1, index=ex_sentence_ids)
    # [batch, path_len * max_sent_len]
    gather_tokens = torch.gather(source, dim=1, index=gathered_sent_token_ids).reshape(batch, path_len, max_sent_len)
    # Union mask
    union_mask = gathered_sent_token_mask & ex_sentence_ids_mask
    return gather_tokens, union_mask


def keep_grad_prompt(input_embeds: Tensor, prompt_pos: Tensor):
    kp_gradient_mask = input_embeds.new_zeros(input_embeds.size()[:-1])  # [batch, seq_len], the position to keep grad is set to ``1``.
    kp_gradient_mask = torch.scatter(kp_gradient_mask, dim=1, index=prompt_pos, value=1.0)
    kp_gradient_mask = kp_gradient_mask.unsqueeze(-1)

    input_embeds_sg = input_embeds.detach()
    input_embeds = kp_gradient_mask * input_embeds + (1 - kp_gradient_mask) * input_embeds_sg
    return input_embeds


def get_accuracy(logits: Tensor, labels: Tensor):
    assert logits.size()[:-1] == labels.size()

    _, pred = logits.max(dim=-1)
    true_label_num = (labels != -1).sum().item()
    correct = (pred == labels).sum().item()
    if true_label_num == 0:
        return 0, 0
    acc = correct * 1.0 / true_label_num
    return acc, true_label_num
