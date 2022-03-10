#for step in 100 200 300 400 500; do
#  export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
#  for seed in 42 43 44 45 4321; do
#    export output_dir=experiments/roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.2.0.w2.2080Ti.s${seed}
#    python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base_v2.py \
#    seed=$seed model_name_or_path=$cp output_dir=$output_dir -cp conf/roberta -cn ft_v8_2_2_1aug_ctx_fix
#  done;
#done;


#for step in 400 500; do
#    export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
#  for seed in 42 43 44 45 4321; do
#    export output_dir=experiments/roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.5.1.w2.2080Ti.et5.s${seed}
#    python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base_v2.py \
#    seed=$seed model_name_or_path=$cp read_tensor.token_num=5 output_dir=$output_dir -cp conf/roberta -cn p_ft_v8_2_2_1aug_ctx_fix
#  done;
#done;


#for step in 100 200 300; do
#    export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
#  for seed in 42 43 44 45 4321; do
#    export output_dir=experiments/roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.5.1.w2.2080Ti.et5.s${seed}
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py \
#    seed=$seed model_name_or_path=$cp read_tensor.token_num=5 output_dir=$output_dir -cp conf/roberta -cn p_ft_v8_2_2_1aug_ctx_fix
#  done;
#done;


# ============================
# extended token num = 0
# ============================
#for step in 100 200 300; do
#    export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
#  for seed in 42 43 44 45 4321; do
#    export output_dir=experiments/roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.5.1.w2.2080Ti.et0.s${seed}
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10085 reclor_trainer_base_v2.py \
#    seed=$seed model_name_or_path=$cp model._target_=models.roberta_baseline.RobertaForMultipleChoiceForZeroShot.from_pretrained \
#    +model.mlp_hidden_size=2048 output_dir=$output_dir -cp conf/roberta -cn ft_v8_2_2_1aug_ctx_fix
#  done;
#done;

##for step in 100 200 300; do
#for step in 400 500; do
#    export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
#  for seed in 42 43 44 45 4321; do
#    export output_dir=experiments/roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.5.0.w2.2080Ti.et0.s${seed}
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10085 reclor_trainer_base_v2.py \
#    seed=$seed model_name_or_path=$cp model._target_=models.roberta_baseline.RobertaForMultipleChoiceForZeroShot.from_pretrained \
#    +model.mlp_hidden_size=2048 output_dir=$output_dir learning_rate=1.5e-5 -cp conf/roberta -cn ft_v8_2_2_1aug_ctx_fix
#  done;
#done;


for step in 100 200 300 400 500; do
  export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
  for seed in 42 43 44 45 4321; do
    export output_dir=experiments/roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.5.0.w2.2080Ti.et0.s${seed}
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10085 reclor_trainer_base_v2.py \
    seed=$seed model_name_or_path=$cp model._target_=models.roberta_baseline.RobertaForMultipleChoiceForZeroShot.from_pretrained \
    +model.mlp_hidden_size=2048 output_dir=$output_dir -cp conf/roberta -cn ft_v8_2_2_1aug_ctx_fix
  done;
done;
