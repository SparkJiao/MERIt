for seed in 42 43 44 45; do
USE_FP16=true python run_squad.py \
  --model_name_or_path pretrained-models/roberta-large \
  --dataset_name squad_v2 \
  --version_2_with_negative true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 24 \
  --weight_decay 0.01 \
  --learning_rate 1.5e-5 \
  --num_train_epochs 2 \
  --num_warmup_steps 330 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --seed $seed \
  --output_dir experiments/squadv2.roberta.large.baseline.v1.0.s$seed
done;