#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=42 -cn ft_v8_2_2_1aug_ctx_1k
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=43 -cn ft_v8_2_2_1aug_ctx_1k
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=44 -cn ft_v8_2_2_1aug_ctx_1k
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=45 -cn ft_v8_2_2_1aug_ctx_1k
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=4321 -cn ft_v8_2_2_1aug_ctx_1k

#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=42 -cn p_ft_v8_2_2_1aug_ctx_1k

#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=43 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=44 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=45 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=4321 -cn p_ft_v8_2_2_1aug_ctx_1k

## TitanXP // 3 GPU // MLP-prompt-tuning
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=$seed -cn p_ft_v8_2_2_1aug_ctx_1k
#done;

## TeslaT4 // 3 GPU // checkpoint step 400
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 3 reclor_trainer_base.py --seed=$seed -cn ft_v8_2_2_1aug_ctx_1k
#done;

# TeslaT4 // 3 GPU
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 3 reclor_trainer_base.py --seed=$seed --read_tensor.token_num=16 -cn p_ft_v8_2_2_1aug_ctx_1k
#done;
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 3 reclor_trainer_base.py --seed=$seed --read_tensor.token_num=5 -cn p_ft_v8_2_2_1aug_ctx_1k
#done;


# TeslaT4 // 2 GPU // MLP-prompt-tuning
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py --seed=$seed -cn p_ft_v8_2_2_1aug_ctx_1k
#done;

# Version v5.2 // learning_rate == 2e-5 // single GPU
#for seed in 42 43 44; do
#for seed in 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cn p_ft_v8_2_2_1aug_ctx_1k
#done;

#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py --seed=${seed} -cn p_ft_v8_2_2_1aug_ctx_1k
#done;


# ========================== Prediction ===============================

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1800 -cn ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1500 -cn ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1000 -cn ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1900 -cn ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1300 -cn ft_v8_2_2_1aug_ctx_1k

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1500 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1500 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1900 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1000 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1600 -cn p_ft_v8_2_2_1aug_ctx_1k

#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base_v2.py seed=${seed} -cp conf/roberta/original -cn ft_et0_v8_2_2_1aug_ctx_1k
#done;

#seed=46
#seed=43
#python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base_v2.py seed=${seed} -cp conf/roberta/original -cn ft_et0_v8_2_2_1aug_ctx_1k

for seed in 42 43 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base_v2.py seed=${seed} -cp conf/roberta -cn p_ft_v8_2_2_1aug_ctx_fix_v3
done;
