#for seed in 42 43 44 45; do
#   deepspeed --include localhost:2,3 reclor_trainer_base_ds_v1.py gradient_accumulation_steps=12 seed=$seed -cp conf/deberta_v2 -cn deberta_ft_baseline
#done;

deepspeed --include localhost:2,3 reclor_trainer_base_ds_v1.py gradient_accumulation_steps=12 seed=46 -cp conf/deberta_v2 -cn deberta_ft_baseline
#deepspeed --include localhost:2,3 reclor_trainer_base_ds_v1.py gradient_accumulation_steps=12 seed=50 -cp conf/deberta_v2 -cn deberta_ft_baseline
