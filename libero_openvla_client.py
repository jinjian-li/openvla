"""
Libero + OpenVLA-7B 评估 — 基座模型基线
本地 Libero 仿真 → 远程 OpenVLA 推理 → OSC_POSE 动作
"""

import os, io, base64, time, json, numpy as np, requests
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SERVER_URL = "http://127.0.0.1:8000/act"
TASK_SUITE = "libero_spatial"
N_EVAL = 10  # 每任务评估轮数

print("=" * 60)
print("Libero + OpenVLA-7B  (Base Model Baseline)")
print("=" * 60)

bench = benchmark.get_benchmark_dict()[TASK_SUITE]()
n_tasks = bench.n_tasks
print(f"  Suite: {TASK_SUITE}, {n_tasks} tasks")

results = []

for task_id in range(n_tasks):
    task = bench.get_task(task_id)
    bddl_file = os.path.join(get_libero_path("bddl_files"), TASK_SUITE, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=224, camera_widths=224,
        has_renderer=False, has_offscreen_renderer=True,
        reward_shaping=True,
        # OSC_POSE 默认 controller — OpenVLA 为此训练
    )
    task_success = 0

    for ep in range(N_EVAL):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done and ep_steps < 200:
            # 截图
            frame = obs["agentview_image"][::-1]  # OpenGL flip
            img = Image.fromarray(frame).resize((224, 224))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            # 远程推理
            try:
                resp = requests.post(
                    SERVER_URL,
                    json={"image": img_b64, "instruction": task.language},
                    timeout=15,
                )
                if resp.status_code == 200:
                    raw = np.array(resp.json()["action"][:7], dtype=float)
                    # Bridge物理单位→OSC_POSE[-1,1]: pos/0.05, rot/0.5, grip/1.0
                    scale = np.array([20, 20, 20, 2, 2, 2, 1.0])
                    action = np.clip(raw * scale, -1, 1)
                else:
                    action = np.zeros(7)
            except:
                action = np.zeros(7)

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

        if done:
            task_success += 1

        print(f"  [{task_id+1}/{n_tasks}] {task.name[:45]:<45} "
              f"ep={ep+1}/{N_EVAL} steps={ep_steps:3d} r={ep_reward:+.3f} done={done}")

    results.append({
        "task": task.name,
        "success": task_success,
        "total": N_EVAL,
        "rate": task_success / N_EVAL,
    })
    env.close()

# 输出
print(f"\n{'='*60}")
print(f"OpenVLA-7B Base — {TASK_SUITE}")
print(f"{'='*60}")
total = 0
for r in results:
    print(f"  {r['task']:<50} {r['success']:>3}/{r['total']}  {r['rate']:>6.1%}")
    total += r['success']
print(f"  {'TOTAL':<50} {total:>3}/{n_tasks*N_EVAL}  {total/(n_tasks*N_EVAL):>6.1%}")

with open("/media/li/新加卷/isaacsim/workspace/openvla_baseline.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results: openvla_baseline.json")
