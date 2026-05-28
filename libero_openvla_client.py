"""
Libero + OpenVLA-7B evaluation client.

Runs local Libero/MuJoCo rollouts and queries a remote OpenVLA FastAPI server.
The default settings are for a base OpenVLA checkpoint using `bridge_orig`
action un-normalization, so the client converts Bridge-style actions into the
Libero OSC_POSE action space.
"""

import argparse
import base64
import io
import json
import os
import time

import numpy as np
import requests
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
    parser.add_argument(
        "--image-transform",
        choices=("rotate180", "vertical", "none"),
        default="rotate180",
        help="Libero agentview preprocessing before sending to OpenVLA.",
    )
    parser.add_argument("--output", default="openvla_baseline_fixed.json")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    return parser.parse_args()


def preprocess_image(obs, image_transform):
    frame = obs["agentview_image"]
    if image_transform == "rotate180":
        # Matches the official OpenVLA Libero eval preprocessing helper.
        frame = frame[::-1, ::-1]
    elif image_transform == "vertical":
        frame = frame[::-1]
    return Image.fromarray(frame).resize((224, 224), Image.Resampling.LANCZOS)


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


def get_initial_states_or_none(bench, task_id):
    try:
        return bench.get_task_init_states(task_id)
    except FileNotFoundError as exc:
        print(f"  init_states missing, falling back to env.reset(): {exc}")
        return None


def main():
    args = parse_args()
    max_steps = get_max_steps(args.task_suite, args.max_steps)

    print("=" * 60)
    print("Libero + OpenVLA-7B  (Base Model Baseline)")
    print("=" * 60)
    print(f"  Server: {args.server_url}")
    print(f"  Suite: {args.task_suite}")
    print(f"  Episodes/task: {args.episodes}")
    print(f"  Image transform: {args.image_transform}")
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
        initial_states = get_initial_states_or_none(bench, task_id)

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=224,
            camera_widths=224,
            has_renderer=False,
            has_offscreen_renderer=True,
            reward_shaping=True,
        )

        task_successes = 0
        task_rewards = []
        task_grips = []

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
                    img = preprocess_image(obs, args.image_transform)
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
                "raw_grip_min": float(np.min(task_grips)) if task_grips else None,
                "raw_grip_max": float(np.max(task_grips)) if task_grips else None,
                "raw_grip_close_ratio": float(np.mean(np.array(task_grips) < 0.5)) if task_grips else None,
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
            "server_url": args.server_url,
            "task_suite": args.task_suite,
            "episodes": args.episodes,
            "task_limit": args.task_limit,
            "max_steps": max_steps,
            "wait_steps": args.wait_steps,
            "image_transform": args.image_transform,
            "action_source": "openvla/openvla-7b bridge_orig",
            "gripper_conversion": "raw < 0.5 => +1 close, raw >= 0.5 => -1 open",
        },
        "results": results,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "total_rate": total_successes / total_episodes if total_episodes else 0.0,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results: {args.output}")


if __name__ == "__main__":
    main()
