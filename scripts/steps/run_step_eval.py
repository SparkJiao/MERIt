import logging
import os
import subprocess
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cmd(command: str):
    logger.info(command)
    subprocess.check_call(command, shell=True)


seed_ls = [42, 43, 44, 45, 4321]
step = [100, 200, 300, 400]
checkpoint_format = "experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.ctx.1k.2080Ti/checkpoint-{}"
output_dir_format = "experiments/roberta.large.wiki_erica_path_v8.2.2.1aug.ctx.1k.2080ti-cp{}.2.0.w1.2080Ti.s{}"
output_dir_format_pt = "experiments/roberta.large.wiki_erica_path_v8.2.2.1aug.ctx.1k.2080ti-cp{}.5.1.w1.2080Ti.s{}"

# for _step in step:
#     for _seed in seed_ls:
#         cmd = f"python reclor_trainer_base.py --seed={_seed} --model_name_or_path={checkpoint_format.format(str(_step))} " \
#               f"--output_dir={output_dir_format.format(str(_step), str(_seed))} -cn ft_v8_2_2_1aug_ctx_1k"
#         run_cmd(cmd)

# for _step in step:
#     for _seed in seed_ls:
#         cmd = f"python reclor_trainer_base.py --seed={_seed} --model_name_or_path={checkpoint_format.format(str(_step))} " \
#               f"--output_dir={output_dir_format_pt.format(str(_step), str(_seed))} -cn p_ft_v8_2_2_1aug_ctx_1k"
#         run_cmd(cmd)

# for _step in [500]:
#     for _seed in seed_ls:
#         cmd = f"python reclor_trainer_base.py --seed={_seed} --model_name_or_path={checkpoint_format.format(str(_step))} " \
#               f"--output_dir={output_dir_format.format(str(_step), str(_seed))} -cn ft_v8_2_2_1aug_ctx_1k"
#         run_cmd(cmd)

# for _step in [500]:
#     for _seed in seed_ls:
#         cmd = f"python reclor_trainer_base.py --seed={_seed} --model_name_or_path={checkpoint_format.format(str(_step))} " \
#               f"--output_dir={output_dir_format_pt.format(str(_step), str(_seed))} -cn p_ft_v8_2_2_1aug_ctx_1k"
#         run_cmd(cmd)


output_dir_format = "experiments/roberta.large.wiki_erica_path_v8.2.2.1aug.ctx.1k.2080ti-cp{}.2.0.w4.2080Ti.s{}"

# for _step in step:
#     for _seed in seed_ls:
#         cmd = f"python -m torch.distributed.launch --nproc_per_node 4 " \
#               f"reclor_trainer_base.py --seed={_seed} --model_name_or_path={checkpoint_format.format(str(_step))} " \
#               f"--gradient_accumulation_steps=6 " \
#               f"--output_dir={output_dir_format.format(str(_step), str(_seed))} -cn ft_v8_2_2_1aug_ctx_1k"
#
#         run_cmd(cmd)

# _step = 100
# for _seed, checkpoint in [(42, 1800), (43, 1200), (44, 1400), (45, 1900), (4321, 1700)]:
# _step = 200
# for _seed, checkpoint in [(42, 1500), (43, 700), (44, 1600), (45, 700), (4321, 1100)]:
# _step = 300
# for _seed, checkpoint in [(42, 1400), (43, 1100), (44, 1600), (45, 1900), (4321, 1400)]:
# _step = 400
# for _seed, checkpoint in [(42, 1300), (43, 1000), (44, 1100), (45, 1400), (4321, 1700)]:
#     cmd = f"python " \
#           f"reclor_trainer_base.py --do_train=False --do_eval=True --eval_sub_path=checkpoint-{checkpoint} " \
#           f"--seed={_seed} --model_name_or_path={checkpoint_format.format(str(_step))} " \
#           f"--gradient_accumulation_steps=6 " \
#           f"--output_dir={output_dir_format.format(str(_step), str(_seed))} -cn ft_v8_2_2_1aug_ctx_1k"
#
#     run_cmd(cmd)


# output_dir_format_pt = "experiments/roberta.large.wiki_erica_path_v8.2.2.1aug.ctx.1k.2080ti-cp{}.5.1.w4.2080Ti.s{}"
#
# for _step in step:
#     for _seed in seed_ls:
#         cmd = f"python -m torch.distributed.launch --nproc_per_node 4 " \
#               f"reclor_trainer_base.py --seed={_seed} --model_name_or_path={checkpoint_format.format(str(_step))} " \
#               f"--gradient_accumulation_steps=6 " \
#               f"--output_dir={output_dir_format_pt.format(str(_step), str(_seed))} -cn p_ft_v8_2_2_1aug_ctx_1k"
#         run_cmd(cmd)

output_dir_format_pt = "experiments/roberta.large.wiki_erica_path_v8.2.2.1aug.ctx.1k.2080ti-cp{}.5.0.w2.2080Ti.s{}"

# for _step in step:
#     for _seed in seed_ls:
#         cmd = f"python -m torch.distributed.launch --nproc_per_node 2 " \
#               f"reclor_trainer_base.py --seed={_seed} --model_name_or_path={checkpoint_format.format(str(_step))} " \
#               f"--gradient_accumulation_steps=12 learning_rate=1e-5 " \
#               f"--output_dir={output_dir_format_pt.format(str(_step), str(_seed))} -cn p_ft_v8_2_2_1aug_ctx_1k"
#         run_cmd(cmd)

# _step = 500
# for _seed in seed_ls:
#     cmd = f"python -m torch.distributed.launch --nproc_per_node 2 " \
#           f"reclor_trainer_base.py seed={_seed} model_name_or_path={checkpoint_format.format(str(_step))} " \
#           f"gradient_accumulation_steps=12 learning_rate=1e-5 " \
#           f"output_dir={output_dir_format_pt.format(str(_step), str(_seed))} -cn p_ft_v8_2_2_1aug_ctx_1k"
#     run_cmd(cmd)


output_dir_format_pt = "experiments/roberta.large.wiki_erica_path_v8.2.2.1aug.ctx.1k.2080ti-cp500.5.1.w2.2080Ti.et{}.s{}"

# for token_num in [5, 15, 20]:
#     for _seed in seed_ls:
#         cmd = f"python -m torch.distributed.launch --nproc_per_node 2 " \
#               f"reclor_trainer_base.py --seed={_seed} --model_name_or_path={checkpoint_format.format(str(500))} " \
#               f"--gradient_accumulation_steps=12 read_tensor.token_num={token_num} " \
#               f"--output_dir={output_dir_format_pt.format(str(token_num), str(_seed))} -cn p_ft_v8_2_2_1aug_ctx_1k"
#         run_cmd(cmd)

token_num = 0
for _seed in seed_ls:
    cmd = f"python -m torch.distributed.launch --nproc_per_node 2 " \
          f"reclor_trainer_base.py --seed={_seed} --model_name_or_path={checkpoint_format.format(str(500))} " \
          f"--gradient_accumulation_steps=12 read_tensor.token_num={token_num} " \
          f"--output_dir={output_dir_format_pt.format(str(token_num), str(_seed))} -cn p_ft_v8_2_2_1aug_ctx_1k"
    run_cmd(cmd)

# output_dir_format_pt = "experiments/roberta.large.wiki_erica_path_v8.2.2.1aug.ctx.1k.2080ti-cp500.5.0.w2.2080Ti.et{}.s{}"
#
# for token_num in [5, 15, 20]:
#     for _seed in seed_ls:
#         cmd = f"python -m torch.distributed.launch --nproc_per_node 2 " \
#               f"reclor_trainer_base.py seed={_seed} model_name_or_path={checkpoint_format.format(str(500))} " \
#               f"gradient_accumulation_steps=12 learning_rate=1e-5 read_tensor.token_num={token_num} " \
#               f"output_dir={output_dir_format_pt.format(str(token_num), str(_seed))} -cn p_ft_v8_2_2_1aug_ctx_1k"
#         run_cmd(cmd)
