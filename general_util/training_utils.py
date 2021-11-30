import random
from typing import Dict
from omegaconf import DictConfig

import numpy as np
import torch


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).
    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def batch_to_device(batch: Dict[str, torch.Tensor], device):
    batch_on_device = {}
    for k, v in batch.items():
        batch_on_device[k] = v.to(device)
    return batch_on_device


def note_best_checkpoint(cfg: DictConfig, results: Dict[str, float], sub_path: str):
    metric = results[cfg.prediction_cfg.metric]
    if (not cfg.prediction_cfg.best_result) or (cfg.prediction_cfg.measure > 0 and metric > cfg.prediction_cfg.best_result) or (
            cfg.prediction_cfg.measure < 0 and metric < cfg.prediction_cfg.best_result):
        cfg.prediction_cfg.best_result = metric
        cfg.prediction_cfg.best_checkpoint = sub_path
