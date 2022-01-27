##### CHECKED
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed -cn ft_v8_2_2_1aug_1k
#done
#
##### CHECKED
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed -cn ft_v8_2_2_1aug_ctx_no_deduct_1k
#done

#sleep 3h 30s

# `seed=44` leads to poor performance.  ##### CHECKED
#python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=46 -cn ft_v8_2_2_1aug_ctx_no_deduct_1k


##### CHECKED
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed read_tensor.token_num=5 -cn p_ft_v8_2_2_1aug_1k
#done

##### CHECKED
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed read_tensor.token_num=5 -cn p_ft_v8_2_2_1aug_ctx_no_deduct_1k
#done

# seed=43 leads to poor performance.  ##### CHECKED
#python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=46 read_tensor.token_num=5 -cn p_ft_v8_2_2_1aug_ctx_no_deduct_1k

##### CHECKED
#python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=4321 read_tensor.token_num=5 -cn p_ft_v8_2_2_1aug_ctx_no_deduct_1k


#for seed in 42 43 44 45 4321; do  ##### CHECKED
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed -cn ft_v8_2_2_simple_1aug_random_noise_num_ctx_1k
#done;
# seed=42 leads to poor performance  ##### CHECKED
python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=46 -cn ft_v8_2_2_simple_1aug_random_noise_num_ctx_1k

#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed read_tensor.token_num=5 -cn p_ft_v8_2_2_2aug_ctx_1k
#done
# seed=42 leads to poor performance.
python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=46 read_tensor.token_num=5 -cn p_ft_v8_2_2_2aug_ctx_1k

#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed read_tensor.token_num=5 -cn p_ft_v8_2_2_3aug_ctx_1k
#done
