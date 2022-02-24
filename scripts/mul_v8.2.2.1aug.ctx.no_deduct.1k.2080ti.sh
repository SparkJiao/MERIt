
#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cn ft_v8_2_2_1aug_ctx_no_deduct_1k
#done;


#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cn p_ft_v8_2_2_1aug_ctx_no_deduct_1k
#done;


# ========================== Prediction ===============================

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-800 -cn ft_v8_2_2_1aug_ctx_no_deduct_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1500 -cn ft_v8_2_2_1aug_ctx_no_deduct_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1600 -cn ft_v8_2_2_1aug_ctx_no_deduct_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-900 -cn ft_v8_2_2_1aug_ctx_no_deduct_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-800 -cn ft_v8_2_2_1aug_ctx_no_deduct_1k

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1900 -cn p_ft_v8_2_2_1aug_ctx_no_deduct_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1900 -cn p_ft_v8_2_2_1aug_ctx_no_deduct_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1200 -cn p_ft_v8_2_2_1aug_ctx_no_deduct_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1600 -cn p_ft_v8_2_2_1aug_ctx_no_deduct_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1600 -cn p_ft_v8_2_2_1aug_ctx_no_deduct_1k


# =============================

for seed in 42 43 44 45 4321; do
  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base_v2.py seed=$seed -cp conf/roberta/original -cn ft_et0_v8_2_2_1aug_ctx_no_deduct_1k
done;
