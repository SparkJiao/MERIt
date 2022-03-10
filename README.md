# MERIt: Meta-Path Guided Contrastive Learning for Logical Reasoning

This is the pytorch implementation of the paper: 

**MERIt: Meta-Path Guided Contrastive Learning for Logical Reasoning.** Fangkai Jiao, Yangyang Guo, Xuemeng Song, Liqiang Nie. _Findings of ACL_. 2022.

[paper link](https://arxiv.org/abs/2203.00357)

## Project Structure

```
|--- conf/   // The configs of all experiments in yaml.
|--- dataset/   // The classes or functions to convert raw text inputs as tensor and utils for batch sampling.
|--- experiments/   // We may provide the prediction results on the datasets, e.g., the .npy file that can be submitted to ReClor leaderboard.
|--- general_util/  // Metrics and training utils.
|--- models/    // The transformers for pre-training and fine-tuning.
|--- modules/
|--- preprocess/    // The scripts to pre-process Wikipedia documents for pre-training.
|--- scripts/   // The bash scripts of our experiments.
|--- reclor_trainer....py   // Trainers for fine-tuning.
|--- trainer_base....py     // Trainers for pre-training.
```

## Requirements

For general requirements, please follow the ``requirements.txt`` file.

Some special notes:
- The [fairscale](https://github.com/facebookresearch/fairscale) is used for fast and memory efficient distributed training, and it's not necessary.
- [NVIDIA Apex](https://github.com/NVIDIA/apex) is required if you want to use the ``FusedLAMB`` optimizer for pre-training. It's also not necessary since you can use ``AdamW``.
- ``RoBERTa-large`` requires at least 12GB memory on single GPU, ``ALBERT-xxlarge`` requires at least 14GB, and we use A100 GPU for ``Deberta-v2-xlarge`` pre-training. Using fairscale can help reduce the memory requirements with more GPUs. We have tried to use ``Deepspeed`` for ``Deberta`` pre-training through CPU-offload but failed.

## Pre-training

The pre-trained models are available at HuggingFace:
- [RoBERTa-large-v1](https://huggingface.co/chitanda/merit-roberta-large-v1)
- [RoBERTa-large-v2](https://huggingface.co/chitanda/merit-roberta-large-v2)
- [ALBERT-v2-xxlarge-v1](https://huggingface.co/chitanda/merit-albert-v2-xxlarge-v1)
- [DeBERTa-v2-xlarge-v1](https://huggingface.co/chitanda/merit-deberta-v2-xlarge-v1)
- [DeBERTa-v2-xxlarge-v1](https://huggingface.co/chitanda/merit-deberta-v2-xxlarge-v1)

For all model checkpoints (including models of different pre-training steps and the models for ablation studies), we will provide them later.

You can also pre-train the models by yourself as following.

### Data Preprocess

Our pre-training procedure uses the data pre-processed by [Qin et al.](https://github.com/thunlp/ERICA), which includes the entities
and the distantly annotated relations in Wikipedia. You can also download it from [here](https://drive.google.com/file/d/1adx2Q6pZ4TYYwk2GUCYeyNtFXc8P2qJF/view?usp=sharing).

To further pre-process the data, run:
```
python preprocess wiki_entity_path_reprocess_v7.py --input_file <glob path for input data> --output_dir <output path>
```
The processed data can also be downloaded from [here](https://drive.google.com/file/d/1pe2p9_P4PoqP2BBTFq38OnNFqWcBnOKy/view?usp=sharing).

### Running

We use [hydra](https://hydra.cc/) to manage the configuration of our experiments. The configs for pre-training are listed below:
```
# RoBERTa-large-v1
conf/wiki_path_large_v8_2_aug1.yaml

# RoBERTa-large-v2
conf/roberta/wiki_path_large_v8_2_aug1_fix_v3.yaml

# ALBERT-v2-xxlarge
conf/wiki_path_albert_v8_2_aug1.yaml

# DeBERTa-v2-xlarge
conf/deberta_v2/wiki_path_deberta_v8_2_aug1.yaml

# DeBERTa-v2-xxlarge
conf/deberta_v2/wiki_path_deberta_v8_2_aug1_xxlarge.yaml
```
To run pre-training:
```
python -m torch.distributed.launch --nproc_per_node N trainer_base_mul_v2.py -cp <directory of config> -cn <name of the config file>
```
DeepSpeed is also supported by using `trainer_base_mul_ds_v1.py` and relevant config is specified in the `ds_config` field of the config.

## Fine-tuning

To run fine-tuning:
```
python -m torch.distributed.launch --nproc_per_node N reclor_trainer_base_v2.py -cp <directory of config> -cn <name of the config file>
```

The configs for different experiments are listed as following (the details of our used GPUs and the seed to achieve the best accuracy on the test set are also provided):

**MERIt-roberta-large-v1**
- ReClor
  - config: conf/ft_v8_2_2_1aug_ctx_1k.yaml (4 * 2080Ti, seed=4321)

[comment]: <> (  - [checkpoint]&#40;&#41;)
- LogiQA
  - config: conf/logiqa/f_logiqa_large_v8_2_1aug_ctx.yaml (1 * Tesla T4, seed=44)

[comment]: <> (  - [checkpoint]&#40;&#41;)

**MERIt-roberta-large-v1 + Prompt**
- ReClor
  - config: conf/p_ft_v8_2_2_1aug_ctx_1k.yaml (2 * 2080Ti, seed=43)

[comment]: <> (  - [checkpoint]&#40;&#41;)
- LogiQA
  - config: conf/logiqa/pf_logiqa_large_v8_2_1aug_ctx.yaml (1 * Tesla T4, seed=44)

[comment]: <> (  - [checkpoint]&#40;&#41;)

**MERIt-roberta-large-v2**
- ReClor
  - config: scripts/steps/run_steps_fix.sh (2 * 2080Ti, seed=43)

[comment]: <> (  - [checkpoint]&#40;&#41;)

[comment]: <> (- LogiQA)

[comment]: <> (  - config:)

[comment]: <> (  - [checkpoint]&#40;&#41;)

**MERIt-roberta-large-v2 + Prompt**
- ReClor
  - config: conf/roberta/p_ft_v8_2_2_1aug_ctx_fix_v3.yaml (2 * 2080Ti, seed=4321)

[comment]: <> (  - [checkpoint]&#40;&#41;)

[comment]: <> (- LogiQA)

[comment]: <> (  - config:)

[comment]: <> (  - [checkpoint]&#40;&#41;)

**MERIt-alberta-v2-xxlarge**
- ReClor
  - config: conf/albert/albert_ft_v822_1aug_ctx (2 * TitanXP, seed=4321)

[comment]: <> (  - [checkpoint]&#40;&#41;)
- LogiQA
  - config: conf/albert/logiqa_albert_ft_v822_1aug_ctx.yaml (1 * 2080Ti, seed=42)

[comment]: <> (  - [checkpoint]&#40;&#41;)


**MERIt-alberta-v2-xxlarge + Prompt**
- ReClor
  - config: conf/albert/albert_pt_v822_1aug_ctx.yaml (2 * TeslaT4, seed=44)

[comment]: <> (  - [checkpoint]&#40;&#41;)
- LogiQA
  - config: conf/albert/logiqa_albert_pt_v822_1aug_ctx.yaml (1 * 2080Ti, seed=43)

[comment]: <> (  - [checkpoint]&#40;&#41;)

**MERIt-deberta-v2-xlarge**
- ReClor
  - config: conf/deberta_v2/deberta_ft_path_v1_4_2.yaml (2 * A100, seed=42)

[comment]: <> (  - [checkpoint]&#40;&#41;)

**MERIt-deberta-v2-xxlarge**
- ReClor
  - config: conf/deberta_v2/deberta_xxlarge_ft_path.yaml (2 * A100, seed=42)

[comment]: <> (  - [checkpoint]&#40;&#41;)


## Citation

If the paper and code are helpful, please kindly cite our paper:
```
@inproceedings{Jiao22merit,
  author    = {Fangkai Jiao and
               Yangyang Guo and
               Xuemeng Song and
               Liqiang Nie},
  title     = {{MERI}t: Meta-Path Guided Contrastive Learning for Logical Reasoning},
  booktitle = {Findings of ACL},
  publisher = {{ACL}},
  year      = {2022}
}
```
