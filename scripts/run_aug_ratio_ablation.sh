# TeslaT4

for seed in 42 43 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed read_tensor.token_num=5 \
  per_gpu_train_batch_size=2 gradient_accumulation_steps=6 \
  output_dir=experiments/roberta.large.wiki_erica_path_v8.2.2.2aug.ctx.1k.2080ti-cp500.5.1.w2.TeslaT4.et5.s${seed} \
   -cn p_ft_v8_2_2_2aug_ctx_1k
done
