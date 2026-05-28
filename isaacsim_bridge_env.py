#!/usr/bin/env python3
"""
Isaac Sim + Bridge-style robot environment.
Headless simulation with Franka/Panda arm for VLA policy rollouts.

Usage:
    python isaacsim_bridge_env.py --task pick_place --episodes 10
"""

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="pick_place", choices=["pick_place", "open_drawer", "wipe_table"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--camera_width", type=int, default=256)
    parser.add_argument("--camera_height", type=int, default=256)
    return parser.parse_args()


class BridgeEnv:
    """Minimal Bridge-style environment wrapper for Isaac Sim."""

    def __init__(self, task: str, headless: bool = True, cam_size: tuple = (256, 256)):
        from isaacsim import SimulationApp

        self._sim = SimulationApp({"headless": headless})
        self._task = task
        self._cam_size = cam_size
        self._setup_scene()

    def _setup_scene(self):
        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicCuboid
        from omni.isaac.manipulators import SingleManipulator
        from omni.isaac.manipulators.grippers import ParallelGripper
        from omni.isaac.core.utils.stage import add_reference_to_stage

        self._world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
        self._world.scene.add_default_ground_plane()

        # Add Franka robot
        add_reference_to_stage(
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Robots/Franka/franka.usd",
            prim_path="/World/Franka",
        )

        # Task-specific objects
        if self._task == "pick_place":
            self._cube = DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.5, 0.1, 0.05]),
                scale=np.array([0.04, 0.04, 0.04]),
                color=np.array([1.0, 0.0, 0.0]),
            )
            self._world.scene.add(self._cube)

        self._world.reset()

    def reset(self) -> dict:
        self._world.reset()
        self._step_count = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple:
        self._apply_action(action)
        self._world.step(render=True)
        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._step_count >= 300
        return obs, reward, done, {}

    def _get_obs(self) -> dict:
        return {
            "image": np.zeros((*self._cam_size, 3), dtype=np.uint8),
            "state": np.zeros(7, dtype=np.float32),
        }

    def _apply_action(self, action: np.ndarray):
        pass

    def _compute_reward(self) -> float:
        return 0.0

    def close(self):
        self._sim.close()


def main():
    args = parse_args()
    print(f"[task] {args.task} | headless={args.headless}")

    env = BridgeEnv(
        task=args.task,
        headless=args.headless,
        cam_size=(args.camera_width, args.camera_height),
    )

    rewards = []
    for ep in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        for step in range(args.max_steps):
            action = np.random.uniform(-0.05, 0.05, size=7)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        print(f"  ep {ep+1}/{args.episodes} | reward {total_reward:.3f}")

    env.close()
    print(f"[done] avg reward: {np.mean(rewards):.3f}")


if __name__ == "__main__":
    main()
