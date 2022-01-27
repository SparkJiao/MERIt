
#for seed in 42 43 44 45 4321; do
#  python -m torch.distributed.launch --nproc_per_node 2 reclor_trainer_base_v2.py seed=$seed -cp conf/albert -cn logiqa_albert_pt_v822_1aug_ctx_et0
#done;

#seed=45
seed=4321
python -m torch.distributed.launch --nproc_per_node 2 --master_port 10001 reclor_trainer_base_v2.py seed=$seed -cp conf/albert -cn logiqa_albert_pt_v822_1aug_ctx_et0

