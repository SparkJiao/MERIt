
#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cp conf/logiqa -cn pf_logiqa_large_v8_2_simple_no_aug_ctx_random_noise_num
#done;


#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cp conf/logiqa -cn pf_logiqa_large_v8_2_simple_1aug_ctx_random_noise_num
#done;


#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cp conf/logiqa -cn f_logiqa_large_v8_2_simple_no_aug_ctx_random_noise_num
#done;

#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py seed=$seed read_tensor.token_num=5 -cp conf/logiqa -cn pf_logiqa_large_v8_2_simple_no_aug_ctx_random_noise_num
#done;

#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py seed=$seed read_tensor.token_num=5 -cp conf/logiqa -cn pf_logiqa_large_v8_2_simple_1aug_ctx_random_noise_num
#done;

for seed in 42 43 44 45 4321; do
  python reclor_trainer_base.py --seed=$seed -cp conf/logiqa -cn f_logiqa_large_v8_2_simple_1aug_ctx_random_noise_num
done;
