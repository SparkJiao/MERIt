#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cp conf/albert -cn albert_pt_v822_1aug_ctx
#done;

# apex
#for seed in 42 43 44 45 4321; do
#for seed in 44 45 46 4321; do
#for seed in 46 47 4321; do
#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_apex.py --seed=$seed -cp conf/albert -cn albert_pt_v822_1aug_ctx
#done;
#python reclor_trainer_apex.py --seed=4321 -cp conf/albert -cn albert_pt_v822_1aug_ctx
#python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py --seed=4321 -cp conf/albert -cn albert_pt_v822_1aug_ctx
python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py --seed=46 -cp conf/albert -cn albert_pt_v822_1aug_ctx

## 2gpu fine-tuning
#for seed in 42 43 44 45 4321; do
#for seed in 44 45 4321; do
#for seed in 46 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=$seed -cp conf/albert -cn albert_ft_v822_1aug_ctx
#done;
#python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base.py seed=46 -cp conf/albert -cn albert_ft_v822_1aug_ctx

# 2gpu prompt tuning
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base.py seed=$seed -cp conf/albert -cn albert_pt_v822_1aug_ctx
#done;

#for seed in 42 43 44 45 4321; do
#  python reclor_trainer_base.py --seed=$seed -cp conf/albert -cn albert_ft_v822_1aug_ctx
#done;

