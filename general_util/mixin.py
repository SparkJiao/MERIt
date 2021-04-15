from collections import defaultdict
from typing import Dict, List

import torch

from general_util.average_meter import LogMetric
from general_util.logger import get_child_logger

logger = get_child_logger("Mixin")


class LogMixin:
    eval_metrics: LogMetric = None

    def init_metric(self, *metric_names):
        self.eval_metrics = LogMetric(*metric_names)

    def get_eval_log(self, reset=False):
        if self.eval_metrics is None:
            logger.warning("The `eval_metrics` attribute hasn't been initialized.")

        results = self.eval_metrics.get_log()

        _eval_metric_log = '\t'.join([f"{k}: {v}" for k, v in results.items()])

        if reset:
            self.eval_metrics.reset()

        return _eval_metric_log, results


class PredictionMixin:
    tensor_dict: Dict[str, List] = defaultdict(list)

    def reset_predict_tensors(self):
        self.tensor_dict = defaultdict(list)

    def concat_predict_tensors(self, **tensors: torch.Tensor):
        for k, v in tensors.items():
            self.tensor_dict[k].extend(v.detach().cpu().tolist())

    def get_predict_tensors(self):
        return self.tensor_dict
