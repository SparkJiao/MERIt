import random
from typing import Dict, List
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


def initialize_optimizer(cfg: DictConfig, grouped_parameters: List[Dict]):
    if "optimizer" in cfg and cfg.optimizer == 'lamb':
        if "bit_training" in cfg and cfg.bit_training:
            from bitsandbytes.optim import LAMB8bit

            optimizer = LAMB8bit(grouped_parameters,
                                 lr=cfg.learning_rate,
                                 betas=eval(cfg.adam_betas),
                                 eps=cfg.adam_epsilon,
                                 max_unorm=cfg.max_grad_norm)
        else:
            from apex.optimizers.fused_lamb import FusedLAMB

            optimizer = FusedLAMB(grouped_parameters,
                                  lr=cfg.learning_rate,
                                  betas=eval(cfg.adam_betas),
                                  eps=cfg.adam_epsilon,
                                  use_nvlamb=(cfg.use_nvlamb if "use_nvlamb" in cfg else False),
                                  max_grad_norm=cfg.max_grad_norm)
    else:
        if "bit_training" in cfg and cfg.bit_training:
            from bitsandbytes.optim import AdamW8bit

            optimizer = AdamW8bit(grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon, betas=(eval(cfg.adam_betas)))
        else:
            from transformers import AdamW

            optimizer = AdamW(grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon, betas=(eval(cfg.adam_betas)))

    return optimizer


def note_best_checkpoint(cfg: DictConfig, results: Dict[str, float], sub_path: str):
    metric = results[cfg.prediction_cfg.metric]
    if (not cfg.prediction_cfg.best_result) or (cfg.prediction_cfg.measure > 0 and metric > cfg.prediction_cfg.best_result) or (
            cfg.prediction_cfg.measure < 0 and metric < cfg.prediction_cfg.best_result):
        cfg.prediction_cfg.best_result = metric
        cfg.prediction_cfg.best_checkpoint = sub_path
        return True
    return False
