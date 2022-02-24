#for step in 100 200; do
#  for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py \
#     cp_step=${step} seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_1
#  done;
#done;


#for step in 100 200; do
#  for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py \
#     model.override_pooler=False cp_step=${step} seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_1
#  done;
#done;


#for step in 100 200; do
#  for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10002 reclor_trainer_base_v2.py \
#    cp_step=${step} seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_2
#  done;
#done;


#for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py \
#    cp_step=200 seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_3
#done;


#for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py \
#    cp_step=200 seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_4
#done;

#for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py \
#    cp_step=200 seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_4_1
#done;


#for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py \
#    cp_step=200 seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_5
#done;


#for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py \
#    cp_step=200 seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_4_2
#done;
#
#for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py \
#    cp_step=200 seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_5_3
#done;


#for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py \
#    cp_step=200 seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_4_3
#done;


#for seed in 42 43 44 45 4321; do
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py \
#    cp_step=200 seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_5_2
#done;


for seed in 42 43 44 45 4321; do
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py \
    cp_step=100 seed=${seed} -cp conf/deberta_v2 -cn deberta_ft_path_v1_4_2
done;
