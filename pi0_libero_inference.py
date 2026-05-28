#!/usr/bin/env python3
"""
Pi0-FAST inference on LIBERO benchmark tasks.
Loads a π0 checkpoint and runs evaluation in Isaac Sim headless mode.

Usage:
    python pi0_libero_inference.py --task LIBERO_Spatial_PickAndPlace --episodes 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./pi0_fast_libero.pt")
    parser.add_argument("--task", default="LIBERO_Spatial_PickAndPlace")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--output", default="./pi0_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def make_env(task_name: str, headless: bool = True):
    """Create LIBERO environment via Isaac Sim."""
    from isaacsim import SimulationApp

    sim = SimulationApp({"headless": headless, "width": 640, "height": 480})

    try:
        import libero
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        print("[!] Install libero: pip install libero")
        sim.close()
        sys.exit(1)

    benchmark = libero.get_benchmark("liberov1")
    task_suite = benchmark.get_task_suite(task_name.split("_")[0].lower())
    task = task_suite.get_task(task_name)
    task_bddl = task_suite.get_task_bddl(task_name)

    env_args = {
        "bddl_file_name": task_bddl,
        "camera_heights": 128,
        "camera_widths": 128,
        "has_renderer": not headless,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "reward_shaping": True,
        "robots": ["Panda"],
    }
    env = OffScreenRenderEnv(**env_args)
    return env, sim


def run_episode(env, model, max_steps: int) -> dict:
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_steps:
        images = obs["agentview_image"]
        state = obs["robot0_joint_pos"]
        input_tensor = {
            "images": torch.from_numpy(images).unsqueeze(0).cuda(),
            "state": torch.from_numpy(state).unsqueeze(0).float().cuda(),
        }
        with torch.inference_mode():
            action = model(input_tensor).cpu().numpy().squeeze(0)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    return {"reward": float(total_reward), "steps": steps, "success": info.get("success", False)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[task] {args.task}")
    print(f"[episodes] {args.episodes}")
    print(f"[device] {torch.cuda.get_device_name(0)}")

    env, sim = make_env(args.task, headless=args.headless)

    print(f"[load] checkpoint {args.checkpoint}")
    model = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.eval()

    results = []
    successes = 0
    for ep in range(args.episodes):
        result = run_episode(env, model, args.max_steps)
        results.append(result)
        if result["success"]:
            successes += 1
        if (ep + 1) % 10 == 0:
            rate = successes / (ep + 1)
            print(f"  ep {ep+1}/{args.episodes} | success rate {rate:.2%}")

    env.close()
    sim.close()

    final_rate = successes / args.episodes
    summary = {
        "task": args.task,
        "episodes": args.episodes,
        "success_rate": final_rate,
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[final] {args.task}: success {final_rate:.2%} ({successes}/{args.episodes})")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
