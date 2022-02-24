#for step in 100 200 300 400; do
#  export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.ctx.1k.2080Ti/checkpoint-${step}
#  for seed in 42 43 44 45 4321; do
#    export output_dir=experiments/roberta.large.wiki_erica_path_v8.2.2.1aug.ctx.1k.2080ti-cp${step}.5.1.w2.2080Ti.et5.s${seed}
#    python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py \
#    seed=$seed model_name_or_path=$cp read_tensor.token_num=5 output_dir=$output_dir -cn p_ft_v8_2_2_1aug_ctx_1k
#  done;
#done;

for step in 100 200 300 400; do
  for seed in 42 43 44 45 4321; do
    python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base_v2.py \
    cp_step=$step seed=$seed -cp conf/roberta/original -cn ft_et0_v8_2_2_1aug_ctx_1k_step
  done;
done;
