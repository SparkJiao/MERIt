##for step in 500 400 300; do
#for step in 200 100; do
#  export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
#  for seed in 42 43 44 45 4321; do
#    export output_dir=experiments/logiqa.roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.5.2.w2.2080Ti.et0.s${seed}
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py \
#    seed=$seed model_name_or_path=$cp  output_dir=$output_dir \
#    -cp conf/logiqa -cn pf_logiqa_roberta_v822_1aug_ctx_fix_et0
#  done;
#done;


#for step in 500 400 300; do
#for step in 200 100; do
#  export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
#  for seed in 42 43 44 45 4321; do
#    export output_dir=experiments/logiqa.roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.2.2.wm20.w2.2080Ti.s${seed}
#    python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py \
#    seed=$seed model_name_or_path=$cp  output_dir=$output_dir \
#    -cp conf/logiqa -cn f_logiqa_large_v822_1aug_ctx_fix
#  done;
#done;

# =============================================== Run on server 129, tmux reclor2

#export step=500
#export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
#for seed in 44 45; do
#  export output_dir=experiments/logiqa.roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.2.2.wm20.w2.2080Ti.s${seed}
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py \
#  seed=$seed model_name_or_path=$cp  output_dir=$output_dir \
#  -cp conf/logiqa -cn f_logiqa_large_v822_1aug_ctx_fix
#done;
#
#export step=400
#export cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
#export seed=46
#export output_dir=experiments/logiqa.roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.2.2.wm20.w2.2080Ti.s${seed}
#python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py \
#seed=$seed model_name_or_path=$cp  output_dir=$output_dir \
#-cp conf/logiqa -cn f_logiqa_large_v822_1aug_ctx_fix

step=200
cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
for seed in 44 45; do
  output_dir=experiments/logiqa.roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.2.2.wm20.w2.2080Ti.s${seed}
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py \
  seed=$seed model_name_or_path=$cp  output_dir=$output_dir \
  -cp conf/logiqa -cn f_logiqa_large_v822_1aug_ctx_fix
done;

step=100
cp=experiments/roberta.large.wiki_erica_path_v7_v8.2.2.1aug.fix.2080Ti/checkpoint-${step}
seed=45
output_dir=experiments/logiqa.roberta.large.wiki_erica_path_v822.1aug.fix.2080ti-cp${step}.2.2.wm20.w2.2080Ti.s${seed}
python -m torch.distributed.launch --nproc_per_node 2 --master_port 10000 reclor_trainer_base_v2.py \
seed=$seed model_name_or_path=$cp  output_dir=$output_dir \
-cp conf/logiqa -cn f_logiqa_large_v822_1aug_ctx_fix

