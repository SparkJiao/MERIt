export TASK_NAME=mnli
#export MODEL_PATH=pretrained-models/roberta-large
#export OUTPUT=experiments/mnli.roberta.large.w1.v1.0
export MODEL_PATH=experiments/roberta.large.wiki_erica_path_v8.1.1.2080Ti/checkpoint-500
export OUTPUT=experiments/mnli.roberta.large.path.w2.v1.0

WANDB_DISABLED="true" python -m torch.distributed.launch --nproc_per_node 2 run_glue.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --fp16 \
  --evaluation_strategy steps \
  --logging_steps 200 \
  --save_steps 200 \
  --learning_rate 2e-5 \
  --weight_decay 0.1 \
  --num_train_epochs 3 \
  --output_dir $OUTPUT \
  --warmup_ratio 0.6
