for seed in 43 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed -cp conf/deberta -cn deberta_ft_v822_1aug_ctx
done;