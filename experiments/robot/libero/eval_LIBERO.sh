#!/bin/bash

echo "running Libero 10"

python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
    --task_suite_name libero_10 \
    --center_crop True \
    --num_trials_per_task 10 \
    --run_id_note "$(date +%Y%m%d_%H%M%S)" \
    --use_wandb True \
    --wandb_project "OpenVLA-Eval" \
    --wandb_entity "a4ar3y-mbzuai" \
