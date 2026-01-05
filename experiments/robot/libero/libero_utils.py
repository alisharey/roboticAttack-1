"""Utils for evaluating policies in LIBERO simulation environments."""

import json
import math
import os
from pathlib import Path

import imageio
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import libero.libero as libero_mod
from libero.libero import get_libero_path, set_libero_default_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def _ensure_libero_paths() -> None:
    env_root = os.environ.get("LIBERO_ROOT")
    if env_root:
        try:
            set_libero_default_path(env_root)
            return
        except Exception as exc:
            print(f"[Warning] failed to set LIBERO_ROOT={env_root}: {exc}")
    try:
        benchmark_root = get_libero_path("benchmark_root")
        if not os.path.exists(benchmark_root):
            fallback_root = os.path.dirname(os.path.abspath(libero_mod.__file__))
            set_libero_default_path(fallback_root)
    except Exception as exc:
        print(f"[Warning] failed to validate LIBERO paths: {exc}")


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    _ensure_libero_paths()
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def _overlay_text(img: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return img
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    x, y = 6, 6
    # Draw a simple shadow for readability.
    draw.multiline_text((x + 1, y + 1), text, fill=(0, 0, 0), spacing=2)
    draw.multiline_text((x, y), text, fill=(255, 255, 255), spacing=2)
    return np.array(pil_img)


def save_rollout_video(
    rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    exp_name="test",
    suite_name=None,
    overlay_texts=None,
    entropy_series=None,
    risk_series=None,
):
    """Saves an MP4 replay of an episode."""
    exp_name_str = str(exp_name) if exp_name else "test"
    if exp_name_str == "origin":
        root_dir = "origin"
    else:
        root_dir = "attack"
    suite_name_str = str(suite_name) if suite_name else "unknown_suite"
    rollout_dir = f"./rollouts/{root_dir}/{suite_name_str}/{DATE_TIME}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=20)
    for i, img in enumerate(rollout_images):
        if overlay_texts is not None and i < len(overlay_texts):
            img = _overlay_text(img, overlay_texts[i])
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    if entropy_series:
        plot_path = mp4_path.replace(".mp4", "_entropy.png")
        ent = np.stack(entropy_series, axis=0)  # (T, K)
        avg = ent.mean(axis=1)
        json_path = mp4_path.replace(".mp4", "_entropy.json")
        payload = [
            {
                "step": int(i),
                "avg": float(avg[i]),
                "dims": [float(v) for v in ent[i]],
            }
            for i in range(ent.shape[0])
        ]
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        num_dims = ent.shape[1]
        fig, axes = plt.subplots(1 + num_dims, 1, figsize=(8, 3 + 2 * num_dims), sharex=True)
        axes[0].plot(avg, color="black", linewidth=2, label="avg")
        if risk_series is not None and len(risk_series) == len(avg):
            axes[0].plot(risk_series, color="red", linestyle="--", label="risk10")
        axes[0].set_title("avg entropy")
        axes[0].set_ylabel("entropy")
        if risk_series is not None and len(risk_series) == len(avg):
            axes[0].legend(loc="best", fontsize="small")
        for k in range(num_dims):
            ax = axes[k + 1]
            ax.plot(ent[:, k], label=f"dim{k}")
            ax.set_title(f"dim{k} entropy")
            ax.set_ylabel("entropy")
            ax.legend(loc="best", fontsize="small")
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved entropy plot at path {plot_path}")
        print(f"Saved entropy metadata at path {json_path}")
        if log_file is not None:
            log_file.write(f"Saved entropy plot at path {plot_path}\n")
            log_file.write(f"Saved entropy metadata at path {json_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
