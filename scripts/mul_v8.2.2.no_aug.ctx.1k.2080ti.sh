

# 2080Ti // 2 GPU
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10085 reclor_trainer_base.py --seed=$seed -cn ft_v8_2_2_no_aug_ctx_1k
#done;

# 2080Ti // 2 GPU
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10085 reclor_trainer_base.py --seed=$seed -cn p_ft_v8_2_2_no_aug_ctx_1k
#done;
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 4 --master_port 10085 reclor_trainer_base.py --seed=$seed -cn p_ft_v8_2_2_no_aug_ctx_1k
#done;

# ========================== Prediction ===============================

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-600 -cn ft_v8_2_2_no_aug_ctx_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1300 -cn ft_v8_2_2_no_aug_ctx_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1200 -cn ft_v8_2_2_no_aug_ctx_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1000 -cn ft_v8_2_2_no_aug_ctx_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1700 -cn ft_v8_2_2_no_aug_ctx_1k

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1600 -cn p_ft_v8_2_2_no_aug_ctx_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1700 -cn p_ft_v8_2_2_no_aug_ctx_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1600 -cn p_ft_v8_2_2_no_aug_ctx_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1300 -cn p_ft_v8_2_2_no_aug_ctx_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1500 -cn p_ft_v8_2_2_no_aug_ctx_1k


for seed in 42 43 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10085 reclor_trainer_base_v2.py seed=$seed -cp conf/roberta/original -cn ft_et0_v8_2_2_no_aug_ctx_1k
done;
