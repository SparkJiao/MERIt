for seed in 42 43 44 45; do
  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base_v2.py seed=$seed -cp conf/race -cn ft
done;