python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=42 -cn ft_v8_2_2_1aug_ctx_1k

python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=43 -cn ft_v8_2_2_1aug_ctx_1k

python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=44 -cn ft_v8_2_2_1aug_ctx_1k

python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=45 -cn ft_v8_2_2_1aug_ctx_1k

python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=4321 -cn ft_v8_2_2_1aug_ctx_1k

#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=42 -cn p_ft_v8_2_2_1aug_ctx_1k

python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=43 -cn p_ft_v8_2_2_1aug_ctx_1k

python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=44 -cn p_ft_v8_2_2_1aug_ctx_1k

python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=45 -cn p_ft_v8_2_2_1aug_ctx_1k

python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=4321 -cn p_ft_v8_2_2_1aug_ctx_1k

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
#
#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1300 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-800 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1200 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1000 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1500 -cn p_ft_v8_2_2_1aug_ctx_1k
