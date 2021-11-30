for seed in 42 43 44 45 4321; do
  python reclor_trainer_base.py --seed=$seed -cp conf/knowledge_probing -cn ft_v8_2_2_1aug_ctx_1k
done;

#for seed in 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cp conf/knowledge_probing -cn p_ft_v8_2_2_1aug_ctx_1k
#done;

#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cp conf/knowledge_probing -cn pm_ft_v8_2_2_1aug_ctx_1k
#done;

#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cp conf/knowledge_probing -cn ft_mnli
#done;

# ========================== Prediction ===============================

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1000 -cp conf/knowledge_probing -cn ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-800 -cp conf/knowledge_probing -cn ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1200 -cp conf/knowledge_probing -cn ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-700 -cp conf/knowledge_probing -cn ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-500 -cp conf/knowledge_probing -cn ft_v8_2_2_1aug_ctx_1k

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1500 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1500 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1900 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1000 -cn p_ft_v8_2_2_1aug_ctx_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1600 -cn p_ft_v8_2_2_1aug_ctx_1k

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1000 -cp conf/knowledge_probing -cn ft_mnli
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1900 -cp conf/knowledge_probing -cn ft_mnli
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1600 -cp conf/knowledge_probing -cn ft_mnli
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-900 -cp conf/knowledge_probing -cn ft_mnli
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-400 -cp conf/knowledge_probing -cn ft_mnli

