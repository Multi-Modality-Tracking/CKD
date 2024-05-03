
SCRIPT=ckd
CONFIG=ckd_4b_dropmae_ckd_tf_cc_mask.25
DATASET=lashertestingset
nohup python tracking/test.py \
    ${SCRIPT} \
    ${CONFIG} \
    --dataset ${DATASET} \
    --threads 4 \
    --num_gpus 1 \
    --vis_gpus 1 \
    >./experiments/${SCRIPT}-${CONFIG}-test${DATASET}.log 2>&1 &

