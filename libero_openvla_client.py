"""
Libero + OpenVLA-7B evaluation client.

Runs local Libero/MuJoCo rollouts and queries a remote OpenVLA FastAPI server.
The default settings are for a base OpenVLA checkpoint using `bridge_orig`
action un-normalization, so the client converts Bridge-style actions into the
Libero OSC_POSE action space.
"""

import argparse
import base64
from datetime import datetime, timezone
import io
import json
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


DEFAULT_SERVER_URL = "http://127.0.0.1:8000/act"
DEFAULT_TASK_SUITE = "libero_spatial"
NOOP_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OpenVLA on Libero through a remote server.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--task-suite", default=DEFAULT_TASK_SUITE)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default="openvla/openvla-7b", help="Remote model identity, recorded only.")
    parser.add_argument("--unnorm-key", default="bridge_orig", help="Remote action un-normalization key, recorded only.")
    parser.add_argument("--server-decode", default="do_sample_false", help="Remote decode mode, recorded only.")
    parser.add_argument(
        "--image-transform",
        choices=("rotate180", "vertical", "none"),
        default="rotate180",
        help="Libero agentview preprocessing before sending to OpenVLA.",
    )
    parser.add_argument("--output", default="openvla_baseline_fixed.json")
    parser.add_argument("--request-timeout", type=float, default=30.0)
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


def summarize_grips(grips):
    if not grips:
        return {
            "raw_grip_min": None,
            "raw_grip_max": None,
            "raw_grip_close_ratio": None,
        }
    grip_array = np.array(grips)
    return {
        "raw_grip_min": float(np.min(grip_array)),
        "raw_grip_max": float(np.max(grip_array)),
        "raw_grip_close_ratio": float(np.mean(grip_array < 0.5)),
    }


def preprocess_image(obs, image_transform, image_size):
    frame = obs["agentview_image"]
    if image_transform == "rotate180":
        # Matches the official OpenVLA Libero eval preprocessing helper.
        frame = frame[::-1, ::-1]
    elif image_transform == "vertical":
        frame = frame[::-1]
    return Image.fromarray(frame).resize((image_size, image_size), Image.Resampling.LANCZOS)


def query_openvla(server_url, image, instruction, timeout):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    resp = requests.post(
        server_url,
        json={"image": img_b64, "instruction": instruction},
        timeout=timeout,
    )
    resp.raise_for_status()
    return np.array(resp.json()["action"][:7], dtype=float)


def bridge_to_libero_osc(raw_action):
    """Convert OpenVLA `bridge_orig` output into Libero OSC_POSE action."""
    action = np.zeros(7, dtype=float)

    # Bridge deltas are in physical units. Libero OSC_POSE expects normalized
    # controller commands in [-1, 1].
    action[:3] = raw_action[:3] * 20.0
    action[3:6] = raw_action[3:6] * 2.0

    # `bridge_orig` gripper is [0, 1], with 0 = close and 1 = open after the
    # OpenVLA dataset convention. Libero expects -1 = open and +1 = close.
    action[6] = 1.0 if raw_action[6] < 0.5 else -1.0

    return np.clip(action, -1.0, 1.0)


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


def main():
    args = parse_args()
    max_steps = get_max_steps(args.task_suite, args.max_steps)
    run_started_at = datetime.now(timezone.utc).isoformat()
    git_metadata = get_git_metadata()
    np.random.seed(args.seed)

    print("=" * 60)
    print("Libero + OpenVLA-7B  (Base Model Baseline)")
    print("=" * 60)
    print(f"  Server: {args.server_url}")
    print(f"  Suite: {args.task_suite}")
    print(f"  Episodes/task: {args.episodes}")
    print(f"  Camera/image: {args.camera_resolution}px -> {args.image_size}px")
    print(f"  Image transform: {args.image_transform}")
    print(f"  Seed: {args.seed}")
    print(f"  Gripper: bridge [0,1] -> Libero binarized [-1,+1]")

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
        task_grips = []
        episode_results = []

        for ep in range(args.episodes):
            obs = env.reset()
            if initial_states is not None:
                obs = env.set_init_state(initial_states[ep % len(initial_states)])
            done = False
            ep_reward = 0.0
            ep_grips = []
            start = time.time()

            for step in range(max_steps + args.wait_steps):
                if step < args.wait_steps:
                    obs, reward, done, info = env.step(NOOP_ACTION)
                    ep_reward += reward
                    continue

                try:
                    img = preprocess_image(obs, args.image_transform, args.image_size)
                    raw = query_openvla(args.server_url, img, task.language, args.request_timeout)
                    action = bridge_to_libero_osc(raw)
                    ep_grips.append(float(raw[6]))
                except Exception as exc:
                    print(f"    inference_error step={step}: {type(exc).__name__}: {exc}")
                    action = NOOP_ACTION.copy()

                obs, reward, done, info = env.step(action)
                ep_reward += reward
                if done:
                    break

            elapsed = time.time() - start
            task_rewards.append(float(ep_reward))
            task_grips.extend(ep_grips)
            total_episodes += 1

            if done:
                task_successes += 1
                total_successes += 1

            grip_summary = "n/a"
            if ep_grips:
                close_ratio = float(np.mean(np.array(ep_grips) < 0.5))
                grip_summary = (
                    f"raw_grip=[{min(ep_grips):+.3f},{max(ep_grips):+.3f}] "
                    f"close_ratio={close_ratio:.1%}"
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
                    **summarize_grips(ep_grips),
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
                **summarize_grips(task_grips),
                "episodes": episode_results,
            }
        )
        env.close()

    print(f"\n{'=' * 60}")
    print(f"OpenVLA-7B Base -- {args.task_suite}")
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
            "seed": args.seed,
            "request_timeout": args.request_timeout,
            "model_id": args.model_id,
            "unnorm_key": args.unnorm_key,
            "server_decode": args.server_decode,
            "action_source": f"{args.model_id} {args.unnorm_key}",
            "position_scale": 20.0,
            "rotation_scale": 2.0,
            "action_clip": [-1.0, 1.0],
            "gripper_conversion": "raw < 0.5 => +1 close, raw >= 0.5 => -1 open",
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
