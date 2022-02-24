
for seed in 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py \
  cp_step=200 seed=$seed -cp conf/deberta_v2 -cn deberta_xxlarge_ft_path
done;


