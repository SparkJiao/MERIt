
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=$seed -cn ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k
#done;
#
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py --seed=$seed -cn p_ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k
#done;

# ========================== Prediction ===============================

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1700 -cn ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1800 -cn ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1100 -cn ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1300 -cn ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1200 -cn ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k
#
python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1300 -cn p_ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k

python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1200 -cn p_ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k

python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1200 -cn p_ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k

python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1700 -cn p_ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k

python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1400 -cn p_ft_v8_2_2_simple_no_aug_random_noise_num_ctx_1k
