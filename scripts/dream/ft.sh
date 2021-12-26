for seed in 42 43 44 45; do
  python reclor_trainer_base_v2.py seed=$seed -cp conf/dream -cn ft
done;

#for seed in 42 43 44 45; do
#  python reclor_trainer_base_v2.py seed=$seed -cp conf/dream -cn ft_path
#done;
