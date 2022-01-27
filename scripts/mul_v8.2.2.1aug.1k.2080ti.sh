
#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cn ft_v8_2_2_1aug_1k
#done;


#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cn p_ft_v8_2_2_1aug_1k
#done;


# ========================== Prediction ===============================

#python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1500 -cn ft_v8_2_2_1aug_1k
#
#python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1600 -cn ft_v8_2_2_1aug_1k
#
#python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-1500 -cn ft_v8_2_2_1aug_1k
#
#python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1200 -cn ft_v8_2_2_1aug_1k
#
#python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-1300 -cn ft_v8_2_2_1aug_1k

python reclor_trainer_base.py --seed=42 --eval_sub_path=checkpoint-1600 -cn p_ft_v8_2_2_1aug_1k

python reclor_trainer_base.py --seed=43 --eval_sub_path=checkpoint-1100 -cn p_ft_v8_2_2_1aug_1k

python reclor_trainer_base.py --seed=44 --eval_sub_path=checkpoint-500 -cn p_ft_v8_2_2_1aug_1k

python reclor_trainer_base.py --seed=45 --eval_sub_path=checkpoint-1300 -cn p_ft_v8_2_2_1aug_1k

python reclor_trainer_base.py --seed=4321 --eval_sub_path=checkpoint-500 -cn p_ft_v8_2_2_1aug_1k
