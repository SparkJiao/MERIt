
#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cp conf/logiqa -cn f_logiqa_large_v8_2_1aug_ctx
#done;

#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py --seed=$seed -cp conf/logiqa -cn pf_logiqa_large_v8_2_1aug_ctx
#done;

# token_num = 5
for seed in 42 43 44 45 4321; do
  python reclor_trainer_base.py seed=$seed read_tensor.token_num=5 -cp conf/logiqa -cn pf_logiqa_large_v8_2_1aug_ctx
done;

## token_num = 15
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed read_tensor.token_num=15 -cp conf/logiqa -cn pf_logiqa_large_v8_2_1aug_ctx
#done;

# prompt tuning + MLP
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed -cp conf/logiqa -cn mpf_logiqa_large_v8_2_1aug_ctx
#done;


# ===================

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1800 -cp conf/logiqa -cn f_logiqa_large_v8_2_1aug_ctx
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-3400 -cp conf/logiqa -cn f_logiqa_large_v8_2_1aug_ctx
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-2800 -cp conf/logiqa -cn f_logiqa_large_v8_2_1aug_ctx
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-4200 -cp conf/logiqa -cn f_logiqa_large_v8_2_1aug_ctx
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-3400 -cp conf/logiqa -cn f_logiqa_large_v8_2_1aug_ctx

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-3200 -cp conf/logiqa -cn pf_logiqa_large_v8_2_1aug_ctx

#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-4600 -cp conf/logiqa -cn pf_logiqa_large_v8_2_1aug_ctx
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-2800 -cp conf/logiqa -cn pf_logiqa_large_v8_2_1aug_ctx
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-2800 -cp conf/logiqa -cn pf_logiqa_large_v8_2_1aug_ctx
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-4600 -cp conf/logiqa -cn pf_logiqa_large_v8_2_1aug_ctx
