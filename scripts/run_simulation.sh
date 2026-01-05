#!/bin/bash
python evaluation_tool/eval_queue_single_four_spec.py \
    --exp_path /l/users/ali.abouzeid/interp/roboticAttack-1/adversarial_patches/simulation/targeted/T-dof1-bc6b6456-c8d7-41e1-9598-91d84f70b278\
    --cudaid 0 \
    --trials 10 \
    --max_concurrent_tasks 2 \
    --task libero_10 libero_object libero_goal libero_spatial \
    --wandb_project "OpenVLA-AttackEval" \
    --wandb_entity "a4ar3y-mbzuai"
