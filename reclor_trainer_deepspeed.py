# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import logging
import os
import random
import sys
from typing import Dict, Union

import deepspeed
import hydra
import numpy as np
import torch
from deepspeed import PipelineEngine
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (AutoTokenizer, PreTrainedTokenizer)
from transformers.deepspeed import HfDeepSpeedConfig

from general_util.logger import setting_logger

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

logger: logging.Logger


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


def save_model(model: Union[torch.nn.Module, PipelineEngine], cfg: DictConfig, output_dir: str, tokenizer: PreTrainedTokenizer = None):
    # Save model checkpoint.
    model.save_checkpoint(output_dir)

    # Save tokenizer and training args.
    if cfg.local_rank in [-1, 0]:
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        OmegaConf.save(cfg, os.path.join(output_dir, "training_config.yaml"))
        logger.info("Saving model checkpoint to %s", output_dir)


def batch_to_device(batch: Dict[str, torch.Tensor], device):
    batch_on_device = {}
    for k, v in batch.items():
        batch_on_device[k] = v.to(device)
    return batch_on_device


def train(cfg, model, tokenizer, continue_from_global_step=0):
    """ Train the model """
    if cfg.local_rank in [-1, 0]:
        _dir_splits = cfg.output_dir.split('/')
        _log_dir = '/'.join([_dir_splits[0], 'runs'] + _dir_splits[1:])
        tb_writer = SummaryWriter(log_dir=_log_dir)
    else:
        tb_writer = None

    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    if cfg.deepspeed_transformer_kernel:
        no_decay = no_decay + [
            'attn_nw', 'attn_nb', 'norm_w', 'norm_b', 'attn_qkvb', 'attn_ob',
            'inter_b', 'output_b'
        ]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=optimizer_grouped_parameters,
                                                  config=cfg.deepspeed_config)

    logger.info(optimizer)

    cfg.per_gpu_train_batch_size = model.train_micro_batch_size_per_gpu()
    cfg.gradient_accumulation_steps = model.gradient_accumulation_steps()

    cfg.train_batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)

    num_examples = 0
    if os.path.exists(cfg.train_file):
        train_files = [cfg.train_file]
    else:
        train_files = list(glob.glob(cfg.train_file))

    logger.info("Pre-loading dataset(s) to count the total steps.")
    for _train_file in train_files:
        _sub_train_dataset, _ = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_train_file)
        num_examples += len(_sub_train_dataset)
        del _sub_train_dataset

    if cfg.local_rank != -1:
        cum_steps = int(num_examples * 1.0 / cfg.train_batch_size / dist.get_world_size())
    else:
        cum_steps = int(num_examples * 1.0 / cfg.train_batch_size)

    if "extended_vocab" in cfg and cfg.extended_vocab:
        model.resize_token_embeddings(model.config.vocab_size + hydra.utils.call(cfg.extended_vocab))

    if cfg.max_steps > 0:
        t_total = cfg.max_steps
        cfg.num_train_epochs = cfg.max_steps // (cum_steps // cfg.gradient_accumulation_steps) + 1
    else:
        t_total = cum_steps // cfg.gradient_accumulation_steps * cfg.num_train_epochs

    # num_warmup_steps = int(t_total * cfg.warmup_proportion) if cfg.warmup_proportion else cfg.warmup_steps

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Num Epochs = %d", cfg.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                cfg.train_batch_size * cfg.gradient_accumulation_steps * (dist.get_world_size() if cfg.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    # logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...", continue_from_global_step)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=cfg.local_rank not in [-1, 0])
    set_seed(cfg)  # Added here for reproducibility (even between python 2 and 3)

    train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    for epoch in train_iterator:

        random.shuffle(train_files)

        for _file_index, _train_file in enumerate(train_files):
            logger.info(f"Loading tensors from {_train_file}")
            _sub_train_dataset, _ = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_train_file)
            _sub_train_sampler = RandomSampler(_sub_train_dataset) if cfg.local_rank == -1 else DistributedSampler(_sub_train_dataset)
            train_dataloader = DataLoader(dataset=_sub_train_dataset, sampler=_sub_train_sampler, batch_size=cfg.train_batch_size,
                                          collate_fn=train_collator, num_workers=cfg.num_workers, pin_memory=True,
                                          prefetch_factor=cfg.prefetch_factor)

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True)
            if cfg.local_rank != -1:
                train_dataloader.sampler.set_epoch(epoch * len(train_files) + _file_index)

            for step, batch in enumerate(epoch_iterator):
                # If training is continued from a checkpoint, fast forward
                # to the state of that checkpoint.
                if global_step < continue_from_global_step:
                    if (step + 1) % cfg.gradient_accumulation_steps == 0:
                        # scheduler.step()  # Update learning rate schedule
                        global_step += 1
                    continue

                model.train()
                batch = batch_to_device(batch, cfg.device)

                outputs = model(**batch)
                loss = outputs["loss"]

                if cfg.n_gpu > 1:
                    loss = loss.mean()

                if cfg.gradient_accumulation_steps > 1:
                    loss = loss / cfg.gradient_accumulation_steps

                model.backward(loss)

                tr_loss += loss
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    # if model.is_gradient_accumulation_boundary():
                    global_step += 1

                    # Log metrics
                    if cfg.local_rank in [-1, 0] and cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
                        # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('lr', model.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / cfg.logging_steps, global_step)
                        logging_loss = tr_loss

                    # Save model checkpoint
                    if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                        output_dir = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                        if cfg.local_rank in [-1, 0] and not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model.save_checkpoint(output_dir)

                    # Evaluation
                    if cfg.evaluate_during_training and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                        if cfg.local_rank in [-1, 0]:
                            results = evaluate(cfg, model, tokenizer, prefix=str(global_step), _split="dev")
                            for key, value in results.items():
                                tb_writer.add_scalar(f"eval/{key}", value, global_step)

                if 0 < cfg.max_steps < global_step:
                    epoch_iterator.close()
                    break

            del _sub_train_dataset
            del _sub_train_sampler
            del train_dataloader

            if 0 < cfg.max_steps < global_step:
                train_iterator.close()
                break

        if 0 < cfg.max_steps < global_step:
            train_iterator.close()
            break

    if cfg.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(cfg, model, tokenizer: PreTrainedTokenizer, prefix="", _split="dev"):
    dataset, features = load_and_cache_examples(cfg, tokenizer, _split=_split)

    # if not os.path.exists(cfg.output_dir) and cfg.local_rank in [-1, 0]:
    #     os.makedirs(cfg.output_dir)
    if not os.path.exists(os.path.join(cfg.output_dir, prefix)):
        os.makedirs(os.path.join(cfg.output_dir, prefix))

    cfg.eval_batch_size = cfg.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly
    eval_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=cfg.eval_batch_size,
                                 collate_fn=eval_collator)
    single_model_gpu = unwrap_model(model)
    single_model_gpu.get_eval_log(reset=True)
    # Eval!
    logger.info("***** Running evaluation {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", cfg.eval_batch_size)
    # Seems FSDP does not need to unwrap the model for evaluating.
    model.eval()
    pred_list = []
    prob_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = batch_to_device(batch, cfg.device)

        with torch.no_grad():
            outputs = model(**batch)
            # logits = outputs["logits"].detach().cpu()
            probs = outputs["logits"].softmax(dim=-1).detach().cpu()
            prob, pred = probs.max(dim=-1)
            pred_list.extend(pred.tolist())
            prob_list.extend(prob.tolist())

    metric_log, results = single_model_gpu.get_eval_log(reset=True)
    logger.info("****** Evaluation Results ******")
    logger.info(f"Global Steps: {prefix}")
    logger.info(metric_log)

    prediction_file = os.path.join(cfg.output_dir, prefix, "eval_predictions.npy")
    np.save(prediction_file, pred_list)
    json.dump(prob_list, open(os.path.join(cfg.output_dir, prefix, "eval_probs.json"), "w"))

    return results


def load_and_cache_examples(cfg, tokenizer: PreTrainedTokenizer, _split="train", _file=None):
    if cfg.local_rank not in [-1, 0] and _split == "train":
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if _file is not None:
        input_file = _file
    elif _split == "train":
        input_file = cfg.train_file
    elif _split == "dev":
        input_file = cfg.dev_file
    elif _split == "test":
        input_file = cfg.test_file
    else:
        raise RuntimeError(_split)

    examples, features, res = hydra.utils.call(cfg.read_tensor, file_path=input_file, tokenizer=tokenizer)

    if cfg.local_rank == 0 and _split == "train":
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if isinstance(res, Dataset):
        return res, features

    dataset = TensorDataset(*res)

    return dataset, features


@hydra.main(config_path="conf/deepspeed", config_name="wiki")
def main(cfg: DictConfig):
    deepspeed.init_distributed(dist_backend="nccl")

    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        # dist.init_process_group(backend='nccl')
        cfg.n_gpu = 1
    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   cfg.local_rank, device, cfg.n_gpu, bool(cfg.local_rank != -1))

    # Set seed
    set_seed(cfg)

    # Load pre-trained model and tokenizer
    # if cfg.local_rank not in [-1, 0]:
    #     dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if cfg.pretrain:
        pretrain_state_dict = torch.load(cfg.pretrain, map_location='cpu')
    else:
        pretrain_state_dict = None

    dschf = HfDeepSpeedConfig(cfg.deepspeed_config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model = hydra.utils.call(cfg.model, cfg.model_name_or_path, state_dict=pretrain_state_dict)

    # if cfg.local_rank == 0:
    #     dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # if cfg.local_rank == -1:  # For FullyShardedDDP, place the model on cpu first.
    #     model.to(cfg.device)

    # logger.info("Training/evaluation parameters %s", OmegaConf.to_yaml(cfg))
    if cfg.local_rank in [-1, 0]:
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))

    # Training
    if cfg.do_train:
        global_step, tr_loss = train(cfg, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if cfg.do_train:
        # Create output directory if needed
        if not os.path.exists(cfg.output_dir) and cfg.local_rank in [-1, 0]:
            os.makedirs(cfg.output_dir)

        logger.info("Saving model checkpoint to %s", cfg.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(cfg.output_dir)
        # save_model(model, cfg, cfg.output_dir)
        model.save_checkpoint(cfg.output_dir)
        if cfg.local_rank == -1 or dist.get_rank() == 0:
            tokenizer.save_pretrained(cfg.output_dir)

            # Good practice: save your training arguments together with the trained model
            # torch.save(cfg, os.path.join(cfg.output_dir, 'training_args.bin'))
            OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_args.yaml"))

    # Test
    results = {}
    if cfg.do_eval and cfg.local_rank in [-1, 0]:
        checkpoints = [cfg.output_dir]
        if cfg.eval_sub_path:
            checkpoints = list(
                os.path.dirname(c) for c in
                sorted(glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "pytorch_model.bin", recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info(" the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            split = "dev"

            model = hydra.utils.call(cfg.model, checkpoint)
            model.to(device)

            if cfg.test_file:
                prefix = 'test-' + prefix
                split = "test"

            result = evaluate(cfg, model, tokenizer, prefix=prefix, _split=split)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    """ Startup command:
        TORCH_EXTENSIONS_DIR="/home/share/jiaofangkai/tmp/torch_extensions" \
        deepspeed --include="localhost:0,1,2,3" reclor_trainer_deepspeed.py --deepspeed=True \
        --deepspeed_config=conf/deepspeed/defaults.json
    """

    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args

    main()
