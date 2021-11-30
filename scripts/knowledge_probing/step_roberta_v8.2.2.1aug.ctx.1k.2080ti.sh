
for step in 100 200 300 400; do
  for seed in 42 43 44 45 4321; do
    python reclor_trainer_base.py seed=$seed \
     model_name_or_path=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.ctx.1k.2080Ti/checkpoint-$step \
     output_dir=experiments/roberta.large.path_v8.2.2.1aug.ctx.1k.cp${step}.kb.no_p.d0.1.0.w1.TeslaT4.s${seed} \
     -cp conf/knowledge_probing -cn ft_v8_2_2_1aug_ctx_1k
  done;
done;

