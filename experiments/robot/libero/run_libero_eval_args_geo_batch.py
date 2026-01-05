"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


import wandb
import torch
import random


# Append repo root so that interpreter can find experiments.robot
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
WHITE_PATCH_DIR = ROOT_DIR / "VLAAttacker" / "white_patch"
if str(WHITE_PATCH_DIR) not in sys.path:
    sys.path.insert(0, str(WHITE_PATCH_DIR))
from appply_random_transform import RandomPatchTransform
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_action_with_entropy,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


# @dataclass
# class GenerateConfig:
#     # fmt: off
#
#     #################################################################################################################
#     # Model-specific parameters
#     #################################################################################################################
#     model_family: str = "openvla"                    # Model family
#     pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
#     load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
#     load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
#
#     center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
#
#     #################################################################################################################
#     # LIBERO environment-specific parameters
#     #################################################################################################################
#     task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
#     num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
#     num_trials_per_task: int = 50                    # Number of rollouts per task
#
#     #################################################################################################################
#     # Utils
#     #################################################################################################################
#     run_id_note: Optional[str] = "spatial1"                # Extra note to add in run ID for logging
#     local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
#
#     use_wandb: bool = False                          # Whether to also log results in Weights & Biases
#     wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
#     wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
#
#     seed: int = 7                                    # Random Seed (for reproducibility)
#
#     # fmt: on

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# @draccus.wrap()
def eval_libero(cfg) -> None:
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cudaid)
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    randomPatchTransform = RandomPatchTransform('cpu',False)
    patch = torch.load(cfg.patchroot)
    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    print(f"Log Path:{str(os.path.join(cfg.local_log_dir, cfg.task_suite_name, '.txt'))}")
    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Log Path:{str(os.path.join(cfg.local_log_dir, cfg.task_suite_name, '.txt'))}")
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_tp, total_fp, total_fn = 0, 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_tp, task_fp, task_fn = 0, 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx]) #

            # Setup
            t = 0
            replay_images = []
            entropy_values = []
            overlay_texts = []
            entropy_series = []
            risk_values = []
            episode_detected = False
            done = False
            det_counts = None
            det_gaps = None
            det_threshold = float(cfg.det_threshold)
            det_window = int(cfg.det_window)
            det_gap_allow = int(cfg.det_gap_allow)
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 193  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 254  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 270  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 505  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 373  # longest training demo has 373 steps
            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size) # TODO: ATTACK Here
                    # img = randomPatchTransform.simulation_paste_patch(img,patch)
                    img = randomPatchTransform.simulation_random_patch(img, patch, geometry=True,colorjitter=False, angle=cfg.angle, shx=cfg.shx, shy=cfg.shy,position=(cfg.x,cfg.y))
                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    action, entropy_mean, entropy_tokens = get_action_with_entropy(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        DEVICE=f"cuda:{cfg.cudaid}",
                        debug=cfg.entropy_debug,
                        step=t,
                    )
                    entropy_values.append(entropy_mean)
                    entropy_series.append(entropy_tokens)
                    if det_counts is None:
                        det_counts = [0] * len(entropy_tokens)
                        det_gaps = [0] * len(entropy_tokens)
                    high_mask = entropy_tokens > det_threshold
                    for i, is_high in enumerate(high_mask):
                        if is_high:
                            det_counts[i] += 1
                            det_gaps[i] = 0
                        else:
                            if det_gaps[i] < det_gap_allow:
                                det_gaps[i] += 1
                            else:
                                det_counts[i] = 0
                                det_gaps[i] = 0
                    detected = any(count >= det_window for count in det_counts)
                    if detected:
                        episode_detected = True
                    token_str = ",".join(f"{v:.3f}" for v in entropy_tokens)
                    overlay_texts.append(f"H={entropy_mean:.3f}\nT=[{token_str}]\nALERT={int(detected)}")
                    window = entropy_values[-10:]
                    risk_values.append(float(sum(window) / len(window)))
                    if cfg.use_wandb:
                        wandb.log({"entropy/step": entropy_mean})

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            print(f"Saving replay video...")
            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
                exp_name=cfg.exp_name,
                suite_name=cfg.task_suite_name,
                overlay_texts=overlay_texts,
                entropy_series=entropy_series,
                risk_series=risk_values,
            )

            # Log current results
            print(f"Success: {done}")
            print(f"Detected: {episode_detected}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"Detected: {episode_detected}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()
            if done:
                if episode_detected:
                    task_fp += 1
                    total_fp += 1
            else:
                if episode_detected:
                    task_tp += 1
                    total_tp += 1
                else:
                    task_fn += 1
                    total_fn += 1

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        task_failures = task_episodes - task_successes
        task_tpr = float(task_tp) / float(task_failures) if task_failures > 0 else 0.0
        task_fpr = float(task_fp) / float(task_successes) if task_successes > 0 else 0.0
        print(f"Current task detection TPR: {task_tpr:.3f}")
        print(f"Current task detection FPR: {task_fpr:.3f}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.write(f"Current task detection TPR: {task_tpr:.3f}\n")
        log_file.write(f"Current task detection FPR: {task_fpr:.3f}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                    f"detection_tpr/{task_description}": task_tpr,
                    f"detection_fpr/{task_description}": task_fpr,
                }
            )

    # Save local log file
    log_file.close()


    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)
    # 追加模式打开文件并添加新内容
    with open(os.path.join(cfg.local_log_dir,cfg.task_suite_name+".txt"), "a") as file:
        total_failures = total_episodes - total_successes
        total_tpr = float(total_tp) / float(total_failures) if total_failures > 0 else 0.0
        total_fpr = float(total_fp) / float(total_successes) if total_successes > 0 else 0.0
        file.write(
            "success_rate/total:"
            f"{float(total_successes) / float(total_episodes)}, "
            f"num_episodes/total:{total_episodes} "
            f"detection_tpr/total:{total_tpr:.3f} "
            f"detection_fpr/total:{total_fpr:.3f} "
            f"position_info:{cfg.angle}_{cfg.shx}_{cfg.shy}_{cfg.x}_{cfg.y}\n"
        )

import argparse
from pathlib import Path
from typing import Optional, Union

def parse_args():
    parser = argparse.ArgumentParser(description="Generate configuration for model training/evaluation")
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    parser.add_argument("--model_family", type=str, default="openvla", help="Model family")
    parser.add_argument("--exp_name", type=str, default=f"libero_object", help="Model family")
    parser.add_argument("--pretrained_checkpoint", type=str, default="openvla/openvla-7b-finetuned-libero-object", help="Pretrained checkpoint path")
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    parser.add_argument("--load_in_4bit", type=bool, default=False)
    parser.add_argument("--center_crop", type=bool, default=True, help="Center crop? (if trained w/ random crop image aug)")

    ################################################################################################# ################
    # LIBERO environment-specific parameters
    #################################################################################################################
    parser.add_argument("--task_suite_name", type=str, default="libero_object", help="Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90")
    parser.add_argument("--num_steps_wait", type=int, default=10, help="Number of steps to wait for objects to stabilize in sim")
    parser.add_argument("--num_trials_per_task", type=int, default=100, help="Number of rollouts per task")

    #################################################################################################################
    # Utils
    #################################################################################################################
    parser.add_argument("--run_id_note", type=str, default=f"test_libero_object", help="Extra note to add in run ID for logging")
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs", help="Local directory for eval logs")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to also log results in Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="LIBERO_simulation_test", help="Name of W&B project to log to (use default!)")
    parser.add_argument("--wandb_entity", type=str, default="taowen_wang-rit", help="Name of entity to log under")
    parser.add_argument("--seed", type=int, default=7, help="Random Seed (for reproducibility)")
    parser.add_argument("--patchroot", type=str, default="/spl_data/tw9146/openvla-main/run/white_patch_attack/a5083c2b-1186-4464-ab9f-1056211a2221/4000/patch.pt", help="")
    parser.add_argument("--x", type=int, default=2, help="")
    parser.add_argument("--y", type=int, default=2, help="")
    parser.add_argument("--angle", type=float, default=2, help="")
    parser.add_argument("--shx", type=float, default=2, help="")
    parser.add_argument("--shy", type=float, default=2, help="")
    parser.add_argument("--cudaid", type=int, default=2, help="")
    parser.add_argument("--det_threshold", type=float, default=0.85, help="Per-dim entropy threshold for detector")
    parser.add_argument("--det_window", type=int, default=20, help="Steps above threshold to trigger detector")
    parser.add_argument("--det_gap_allow", type=int, default=1, help="Allowed single-step dips for detector")
    parser.add_argument("--entropy_debug", type=bool, default=False, help="Print action_dim/token counts per step")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cudaid)
    eval_libero(args)
