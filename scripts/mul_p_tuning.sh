
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=42 -cn config_p_tuning
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=43 -cn config_p_tuning
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=44 -cn config_p_tuning
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=45 -cn config_p_tuning
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=4321 -cn config_p_tuning

#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=44 -cn config
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=45 -cn config
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=4321 -cn config


# Predict

CUDA_VISIBLE_DEVICES=0 python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1000 -cn config_p_tuning

CUDA_VISIBLE_DEVICES=0 python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1400 -cn config_p_tuning

CUDA_VISIBLE_DEVICES=0 python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1500 -cn config_p_tuning

CUDA_VISIBLE_DEVICES=0 python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1900 -cn config_p_tuning

CUDA_VISIBLE_DEVICES=0 python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1900 -cn config_p_tuning
