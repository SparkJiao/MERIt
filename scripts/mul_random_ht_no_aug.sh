#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=42 -cn config_random_ht_no_aug
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=43 -cn config_random_ht_no_aug
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=44 -cn config_random_ht_no_aug
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=45 -cn config_random_ht_no_aug
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=4321 -cn config_random_ht_no_aug  # Low accuracy

#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=42 -cn config_random_ht_no_aug_p_tuning
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=43 -cn config_random_ht_no_aug_p_tuning
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=44 -cn config_random_ht_no_aug_p_tuning
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=45 -cn config_random_ht_no_aug_p_tuning
#
#python -m torch.distributed.launch --nproc_per_node 4 reclor_trainer_base.py --seed=4321 -cn config_random_ht_no_aug_p_tuning

# ============== Prediction ===============================

python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1700 -cn config_random_ht_no_aug

python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1100 -cn config_random_ht_no_aug

python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1100 -cn config_random_ht_no_aug

python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1500 -cn config_random_ht_no_aug

python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-900 -cn config_random_ht_no_aug
#
#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-700 -cn config_random_ht_no_aug_p_tuning
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-900 -cn config_random_ht_no_aug_p_tuning
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1600 -cn config_random_ht_no_aug_p_tuning
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-900 -cn config_random_ht_no_aug_p_tuning
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1300 -cn config_random_ht_no_aug_p_tuning

