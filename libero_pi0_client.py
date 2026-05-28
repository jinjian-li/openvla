"""
Libero + PI0-FAST 客户端 — 本地仿真 + 远程 PI0 推理
"""

import os, io, base64, time, numpy as np
import requests
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SERVER_URL = "http://127.0.0.1:6009/act"
TASK_SUITE = "libero_spatial"
N_EVAL = 1

# PI0 反归一化输出 → OSC_POSE [-1,1]，用 mean/std (比 q01/q99 更宽松)
_ACT_MEAN = [0.102, 0.053, -0.852, -2.972, 0.226, 0.120, -0.046]
_ACT_STD  = [0.371, 0.406, 0.623, 0.349, 0.909, 0.345, 0.999]


def normalize_pi0_action(raw_action):
    """z-score 归一化：95% 的值落 [-1,1]"""
    a = np.array(raw_action[:7], dtype=float)
    return np.clip((a - _ACT_MEAN) / (2.0 * np.array(_ACT_STD)), -1.0, 1.0)

print("=" * 60)
print("Libero + PI0-FAST  (Base Model)")
print("=" * 60)

bench = benchmark.get_benchmark_dict()[TASK_SUITE]()
n_tasks = bench.n_tasks
print(f"  Suite: {TASK_SUITE}, {n_tasks} tasks")
print(f"  PI0 Server: {SERVER_URL}")

results = []

for task_id in range(n_tasks):
    task = bench.get_task(task_id)
    bddl_file = os.path.join(get_libero_path("bddl_files"), TASK_SUITE, task.bddl_file)

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 224,
        "camera_widths": 224,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "render_camera": "agentview",
        "reward_shaping": True,
        # OSC_POSE 是 Libero 默认 — PI0 就是在这个 controller 下训练的
    }

    env = OffScreenRenderEnv(**env_args)
    task_success = 0

    for ep in range(N_EVAL):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done and ep_steps < 200:
            img_arr = obs["agentview_image"]
            # Libero agentview_image 已是 RGB，直接 resize 给 PI0
            frame = img_arr[::-1]  # OpenGL 翻转
            img = Image.fromarray(frame).resize((224, 224))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            try:
                # 读取当前本体状态发给 PI0
                robot = env.env.robots[0]
                joint_pos = robot._joint_positions[:7]
                gripper_pos = [robot.controller.gripper_qpos if hasattr(robot.controller, 'gripper_qpos') else 0.0]
                state = list(joint_pos) + gripper_pos

                resp = requests.post(SERVER_URL,
                                   json={"image": img_b64, "instruction": task.language, "state": state},
                                   timeout=15)
                if resp.status_code == 200:
                    action = normalize_pi0_action(resp.json()["action"])
                else:
                    action = np.zeros(7)
            except:
                action = np.zeros(7)

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

        if done:
            task_success += 1

        print(f"  [{task_id+1}/{n_tasks}] task={task.name[:40]:<40} "
              f"ep={ep+1}/{N_EVAL} steps={ep_steps:3d} reward={ep_reward:+.3f} done={done}")

    results.append({
        "task": task.name,
        "success": task_success,
        "total": N_EVAL,
        "rate": task_success / N_EVAL,
    })

    env.close()

# 输出结果
print(f"\n{'='*60}")
print(f"PI0-FAST Base Model — {TASK_SUITE} Results")
print(f"{'='*60}")
print(f"{'Task':<50} {'Success':>8} {'Rate':>8}")
print(f"{'-'*66}")
total_success = 0
for r in results:
    print(f"{r['task']:<50} {r['success']:>3}/{r['total']:<4} {r['rate']:>7.1%}")
    total_success += r['success']
print(f"{'-'*66}")
print(f"{'TOTAL':<50} {total_success:>3}/{len(results)*N_EVAL:<4} {total_success/(len(results)*N_EVAL):>7.1%}")

import json
with open("/media/li/新加卷/isaacsim/workspace/pi0_baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: pi0_baseline_results.json")
