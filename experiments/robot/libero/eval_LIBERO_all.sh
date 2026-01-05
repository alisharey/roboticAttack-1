#!/bin/bash
set -euo pipefail

datasets=("libero_10" "libero_object" "libero_goal" "libero_spatial")

checkpoint_for() {
  case "$1" in
    libero_10) echo "openvla/openvla-7b-finetuned-libero-10" ;;
    libero_object) echo "openvla/openvla-7b-finetuned-libero-object" ;;
    libero_goal) echo "openvla/openvla-7b-finetuned-libero-goal" ;;
    libero_spatial) echo "openvla/openvla-7b-finetuned-libero-spatial" ;;
    *) echo "Unknown dataset: $1" >&2; exit 1 ;;
  esac
}

run_one() {
  local dataset="$1"
  local ckpt
  ckpt="$(checkpoint_for "$dataset")"
  echo "running ${dataset}"
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "${ckpt}" \
    --task_suite_name "${dataset}" \
    --center_crop True \
    --num_trials_per_task 10 \
    --run_id_note "$(date +%Y%m%d_%H%M%S)_${dataset}" \
    --use_wandb True \
    --wandb_project "OpenVLA-Eval" \
    --wandb_entity "a4ar3y-mbzuai" \
    --det_threshold 0.85 \
    --det_window 20 \
    --det_gap_allow 1
}

max_jobs=2
active=0
for dataset in "${datasets[@]}"; do
  run_one "${dataset}" &
  active=$((active + 1))
  if [ "${active}" -ge "${max_jobs}" ]; then
    wait -n
    active=$((active - 1))
  fi
done

wait
