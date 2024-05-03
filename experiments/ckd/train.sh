

SCRIPT=ckd
CONFIG=ckd_4b_dropmae_ckd_tf_cc_mask.25
nohup python tracking/train.py \
    --script ${SCRIPT} \
    --config ${CONFIG} \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 2 \
    --use_wandb 0 \
    --use_lmdb 0 \
    --vis_gpus 0,1 \
    >./experiments/${SCRIPT}-${CONFIG}-train.log 2>&1 &