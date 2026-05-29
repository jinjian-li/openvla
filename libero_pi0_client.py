"""
Libero + PI0-FAST evaluation client.

Runs local Libero/MuJoCo rollouts and queries a remote PI0-FAST FastAPI
server. The remote OpenPI policy already applies its output transforms and
returns Libero-ready OSC_POSE actions, so this client does not renormalize
actions locally.
"""

import argparse
import base64
from collections import deque
from datetime import datetime, timezone
import io
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import requests
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image


DEFAULT_SERVER_URL = "http://127.0.0.1:6009/act"
DEFAULT_TASK_SUITE = "libero_spatial"
NOOP_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PI0-FAST on Libero through a remote server.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--task-suite", default=DEFAULT_TASK_SUITE)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default="pi0_fast_libero", help="Remote policy identity, recorded only.")
    parser.add_argument("--output", default="pi0_baseline_fixed.json")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--clip-actions", action="store_true", help="Clip server actions to [-1, 1] before env.step.")
    parser.add_argument(
        "--image-transform",
        choices=("rotate180", "vertical", "none"),
        default="rotate180",
        help="Libero image preprocessing before sending to PI0.",
    )
    return parser.parse_args()


def get_git_metadata():
    def git_output(*args):
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    tracked_status = git_output("status", "--short", "--untracked-files=no") or ""
    untracked = git_output("ls-files", "--others", "--exclude-standard") or ""
    return {
        "commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "tracked_dirty": bool(tracked_status),
        "tracked_status_short": tracked_status.splitlines(),
        "untracked_files": untracked.splitlines(),
    }


def get_max_steps(task_suite, override):
    if override is not None:
        return override
    defaults = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    return defaults.get(task_suite, 220)


def load_init_states(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def get_initial_states_or_none(bench, task_id):
    task = bench.get_task(task_id)
    candidate_paths = [
        (
            "configured",
            os.path.join(
                get_libero_path("init_states"),
                task.problem_folder,
                task.init_states_file,
            ),
        ),
        (
            "bundled",
            os.path.join(
                get_libero_path("benchmark_root"),
                "init_files",
                task.problem_folder,
                task.init_states_file,
            ),
        ),
    ]

    for source, path in candidate_paths:
        if os.path.exists(path):
            print(f"  init_states using {source} file: {path}")
            states = load_init_states(path)
            return states, source, path

    missing_paths = ", ".join(path for _, path in candidate_paths)
    print(f"  init_states missing, falling back to env.reset(): {missing_paths}")
    return None, "env_reset", None


def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=float).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def get_pi0_state(obs):
    """Return the 8D Libero state expected by OpenPI's Libero policy."""
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def preprocess_frame(frame, image_transform, image_size):
    if image_transform == "rotate180":
        frame = frame[::-1, ::-1]
    elif image_transform == "vertical":
        frame = frame[::-1]
    return Image.fromarray(np.ascontiguousarray(frame)).resize((image_size, image_size), Image.Resampling.LANCZOS)


def encode_png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def query_pi0(server_url, image, wrist_image, instruction, state, timeout):
    resp = requests.post(
        server_url,
        json={
            "image": encode_png(image),
            "wrist_image": encode_png(wrist_image),
            "instruction": instruction,
            "state": state.tolist(),
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "actions" in payload:
        return np.asarray(payload["actions"], dtype=float)
    return np.asarray([payload["action"]], dtype=float)


def summarize_actions(actions):
    if not actions:
        return {
            "action_min": None,
            "action_max": None,
            "gripper_min": None,
            "gripper_max": None,
        }
    arr = np.asarray(actions, dtype=float)
    return {
        "action_min": np.min(arr, axis=0).tolist(),
        "action_max": np.max(arr, axis=0).tolist(),
        "gripper_min": float(np.min(arr[:, 6])),
        "gripper_max": float(np.max(arr[:, 6])),
    }


def main():
    args = parse_args()
    max_steps = get_max_steps(args.task_suite, args.max_steps)
    run_started_at = datetime.now(timezone.utc).isoformat()
    git_metadata = get_git_metadata()
    np.random.seed(args.seed)

    print("=" * 60)
    print("Libero + PI0-FAST  (Base Model)")
    print("=" * 60)
    print(f"  Server: {args.server_url}")
    print(f"  Suite: {args.task_suite}")
    print(f"  Episodes/task: {args.episodes}")
    print(f"  Camera/image: {args.camera_resolution}px -> {args.image_size}px")
    print(f"  Image transform: {args.image_transform}")
    print(f"  Replan steps: {args.replan_steps}")
    print(f"  Seed: {args.seed}")
    print("  State: eef_pos + eef_axis_angle + gripper_qpos")
    print("  Action: server output used directly as Libero OSC_POSE")

    bench = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = bench.n_tasks if args.task_limit is None else min(args.task_limit, bench.n_tasks)
    print(f"  Tasks: {n_tasks}/{bench.n_tasks}")

    results = []
    total_successes = 0
    total_episodes = 0

    for task_id in range(n_tasks):
        task = bench.get_task(task_id)
        bddl_file = os.path.join(get_libero_path("bddl_files"), args.task_suite, task.bddl_file)
        initial_states, init_source, init_path = get_initial_states_or_none(bench, task_id)
        init_count = len(initial_states) if initial_states is not None else 0

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=args.camera_resolution,
            camera_widths=args.camera_resolution,
            has_renderer=False,
            has_offscreen_renderer=True,
            reward_shaping=True,
        )
        env.seed(args.seed)

        task_successes = 0
        task_rewards = []
        task_actions = []
        episode_results = []

        for ep in range(args.episodes):
            obs = env.reset()
            if initial_states is not None:
                obs = env.set_init_state(initial_states[ep % len(initial_states)])
            done = False
            ep_reward = 0.0
            ep_actions = []
            action_plan = deque()
            start = time.time()

            for step in range(max_steps + args.wait_steps):
                if step < args.wait_steps:
                    obs, reward, done, info = env.step(NOOP_ACTION)
                    ep_reward += reward
                    continue

                try:
                    if not action_plan:
                        img = preprocess_frame(obs["agentview_image"], args.image_transform, args.image_size)
                        wrist_frame = obs.get("robot0_eye_in_hand_image")
                        if wrist_frame is None:
                            wrist_frame = np.zeros_like(obs["agentview_image"])
                        wrist_img = preprocess_frame(wrist_frame, args.image_transform, args.image_size)
                        state = get_pi0_state(obs)
                        action_chunk = query_pi0(
                            args.server_url,
                            img,
                            wrist_img,
                            task.language,
                            state,
                            args.request_timeout,
                        )
                        if len(action_chunk) < args.replan_steps:
                            raise ValueError(
                                f"server returned {len(action_chunk)} actions, need {args.replan_steps}"
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = np.asarray(action_plan.popleft()[:7], dtype=float)
                    if args.clip_actions:
                        action = np.clip(action, -1.0, 1.0)
                    ep_actions.append(action.tolist())
                except Exception as exc:
                    print(f"    inference_error step={step}: {type(exc).__name__}: {exc}")
                    action = NOOP_ACTION.copy()

                obs, reward, done, info = env.step(action)
                ep_reward += reward
                if done:
                    break

            elapsed = time.time() - start
            task_rewards.append(float(ep_reward))
            task_actions.extend(ep_actions)
            total_episodes += 1

            if done:
                task_successes += 1
                total_successes += 1

            action_summary = summarize_actions(ep_actions)
            grip_summary = "n/a"
            if action_summary["gripper_min"] is not None:
                grip_summary = (
                    f"grip=[{action_summary['gripper_min']:+.3f},"
                    f"{action_summary['gripper_max']:+.3f}]"
                )

            episode_results.append(
                {
                    "episode": ep,
                    "success": int(bool(done)),
                    "reward": float(ep_reward),
                    "steps": int(step + 1),
                    "elapsed_sec": float(elapsed),
                    "init_state_source": init_source,
                    "init_state_index": int(ep % init_count) if init_count else None,
                    **action_summary,
                }
            )

            print(
                f"  [{task_id+1}/{n_tasks}] {task.name[:45]:<45} "
                f"ep={ep+1}/{args.episodes} r={ep_reward:+.3f} done={done} "
                f"steps={step+1:3d} t={elapsed:.1f}s {grip_summary}"
            )

        rate = task_successes / args.episodes
        results.append(
            {
                "task": task.name,
                "success": task_successes,
                "total": args.episodes,
                "rate": rate,
                "avg_reward": float(np.mean(task_rewards)) if task_rewards else 0.0,
                "init_state_source": init_source,
                "init_state_path": init_path,
                "init_state_count": init_count,
                **summarize_actions(task_actions),
                "episodes": episode_results,
            }
        )
        env.close()

    print(f"\n{'=' * 60}")
    print(f"PI0-FAST Base -- {args.task_suite}")
    print(f"{'=' * 60}")
    for result in results:
        print(f"  {result['task']:<50} {result['success']:>3}/{result['total']}  {result['rate']:>6.1%}")
    print(f"  {'TOTAL':<50} {total_successes:>3}/{total_episodes}  {total_successes / total_episodes:>6.1%}")

    payload = {
        "config": {
            "run_started_at": run_started_at,
            "argv": sys.argv,
            "git": git_metadata,
            "server_url": args.server_url,
            "task_suite": args.task_suite,
            "episodes": args.episodes,
            "task_limit": args.task_limit,
            "max_steps": max_steps,
            "wait_steps": args.wait_steps,
            "camera_resolution": args.camera_resolution,
            "image_size": args.image_size,
            "image_transform": args.image_transform,
            "image_resize": "PIL.Image.Resampling.LANCZOS",
            "replan_steps": args.replan_steps,
            "seed": args.seed,
            "request_timeout": args.request_timeout,
            "model_id": args.model_id,
            "state_format": "robot0_eef_pos + quat2axisangle(robot0_eef_quat) + robot0_gripper_qpos",
            "action_source": "server actions after OpenPI output transforms",
            "clip_actions": args.clip_actions,
        },
        "results": results,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "total_rate": total_successes / total_episodes if total_episodes else 0.0,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  Results: {args.output}")


if __name__ == "__main__":
    main()
