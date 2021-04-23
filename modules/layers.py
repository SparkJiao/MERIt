from torch import Tensor


def get_accuracy(logits: Tensor, labels: Tensor):
    assert logits.size()[:-1] == labels.size()

    _, pred = logits.max(dim=-1)
    true_label_num = (labels != -1).sum().item()
    correct = (pred == labels).sum().item()
    if true_label_num == 0:
        return 0, 0
    acc = correct * 1.0 / true_label_num
    return acc, true_label_num
