for seed in 43 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed -cn ft_mnli
done;