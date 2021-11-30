
#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py seed=$seed -cp conf/knowledge_probing -cn albert_ft_baseline
#done;


#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py seed=$seed -cp conf/knowledge_probing -cn albert_ft_v822_1aug_ctx
#done;


# ========================== Prediction ===============================

python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1800 -cp conf/knowledge_probing -cn albert_ft_baseline

python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1600 -cp conf/knowledge_probing -cn albert_ft_baseline

python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1300 -cp conf/knowledge_probing -cn albert_ft_baseline

python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1200 -cp conf/knowledge_probing -cn albert_ft_baseline

python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1100 -cp conf/knowledge_probing -cn albert_ft_baseline

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1500 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1500 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1900 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1000 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1600 -cn p_ft_v8_2_2_1aug_ctx_1k
