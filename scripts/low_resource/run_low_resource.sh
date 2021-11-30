export token_num=5

#for ratio in 20 40 60 80; do
#  export RECLOR_DIR=/home/share/jiaofangkai/reclor_data/sub-data/reclor-data-${ratio}
#  for seed in 42 43 44 45 4321; do
#    HYDRA_FULL_ERROR=1 python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py \
#      seed=$seed read_tensor.token_num=$token_num train_file=$RECLOR_DIR/train.json \
#      dev_file=$RECLOR_DIR/val.json \
#      test_file=$RECLOR_DIR/test.json \
#      output_dir=experiments/roberta.large.path_v8.2.2.1aug.ctx.1k.2080ti-cp500.5.1.w2.2080Ti.et$token_num.lr$ratio.s$seed \
#      -cn p_ft_v8_2_2_1aug_ctx_1k
#  done;
#done;
export ratio=60
export RECLOR_DIR=/home/share/jiamengzhao/reclor_data/sub-data/reclor-data-60-46
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py \
#      seed=$seed read_tensor.token_num=$token_num train_file=$RECLOR_DIR/train.json \
#      dev_file=$RECLOR_DIR/val.json \
#      test_file=$RECLOR_DIR/test.json \
#      output_dir=experiments/roberta.large.path_v8.2.2.1aug.ctx.1k.2080ti-cp500.5.1.w2.2080Ti.et$token_num.lr${ratio}_46.s${seed} \
#      -cn p_ft_v8_2_2_1aug_ctx_1k
#done;

# roberta-large
#for ratio in 20 40 60 80; do
#  export RECLOR_DIR=/home/share/jiaofangkai/reclor_data/sub-data/reclor-data-${ratio}
#  for seed in 42 43 44 45 4321; do
#    HYDRA_FULL_ERROR=1 python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py \
#      seed=$seed train_file=$RECLOR_DIR/train.json \
#      dev_file=$RECLOR_DIR/val.json \
#      test_file=$RECLOR_DIR/test.json \
#      output_dir=experiments/roberta.large.w2.2.0.2080Ti.lr$ratio.s$seed \
#      -cn config_large
#  done;
#done;

#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py \
#      seed=$seed train_file=$RECLOR_DIR/train.json \
#      dev_file=$RECLOR_DIR/val.json \
#      test_file=$RECLOR_DIR/test.json \
#      output_dir=experiments/roberta.large.w2.2.0.2080Ti.lr${ratio}_46.s${seed} \
#      -cn config_large
#done;

# `Seed=46` leads to poor performance.
#python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py \
#  seed=42 train_file=$RECLOR_DIR/train.json \
#  dev_file=$RECLOR_DIR/val.json \
#  test_file=$RECLOR_DIR/test.json \
#  output_dir=experiments/roberta.large.w2.2.0.2080Ti.lr${ratio}_46.s${seed}.TeslaT4
#  -cn config_large
python reclor_trainer_base.py \
  seed=42 train_file=$RECLOR_DIR/train.json \
  dev_file=$RECLOR_DIR/val.json \
  test_file=$RECLOR_DIR/test.json \
  output_dir=experiments/roberta.large.w1.2.0.2080Ti.lr${ratio}_46.s42.2080Ti \
  gradient_accumulation_steps=24 \
  -cn config_large


#for seed in 42 43 44 45 4321; do
#    HYDRA_FULL_ERROR=1 python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py \
#      seed=$seed  \
#      output_dir=experiments/roberta.large.w2.2.0.2080Ti.s$seed \
#      -cn config_large
#done;


#python reclor_trainer_base.py seed=4321 output_dir=experiments/roberta.large.w2.2.0.2080Ti.lr60.s4321 do_train=False eval_sub_path=checkpoint-700 -cn config_large
