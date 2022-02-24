
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py seed=$seed -cp conf/deberta_v2 -cn deberta_xxlarge_ft_baseline
#done;


for seed in 42 43 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py seed=$seed -cp conf/deberta_v2 -cn deberta_xxlarge_ft_baseline_v1_1
done;

