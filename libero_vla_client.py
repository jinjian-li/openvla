"""
Libero + 远程 OpenVLA 客户端
本地 MuJoCo Libero 仿真 → 截图 → 远程推理 → 动作控制
"""

import os, io, base64, time, subprocess, socket, numpy as np
import requests
from PIL import Image

# === 远程 VLA ===
SERVER_URL = "http://127.0.0.1:8000/act"
SSH_PASSWORD = os.getenv("AUTODL_SSH_PASSWORD")
SSH_TUNNEL_CMD = [
    "sshpass", "-p", SSH_PASSWORD or "",
    "ssh", "-p", "54812", "-o", "StrictHostKeyChecking=no",
    "-N", "-L", "8000:127.0.0.1:6008",
    "root@connect.westc.seetacloud.com",
]


def setup_tunnel():
    if not SSH_PASSWORD:
        raise RuntimeError("Missing AUTODL_SSH_PASSWORD environment variable.")
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", 8000)); s.close(); return True
    except:
        s.close()
        print("  Starting SSH tunnel...")
        subprocess.Popen(SSH_TUNNEL_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2); return False


print("=" * 60)
print("Libero + Remote OpenVLA")
print("=" * 60)

# === 1. 加载 Libero 环境 ===
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark

bench = benchmark.get_benchmark_dict()["libero_spatial"]()
task = bench.get_task(0)
bddl_file = task.bddl_file
# task.bddl_file 是相对路径，需要加完整前缀
from libero.libero import get_libero_path
bddl_file = os.path.join(get_libero_path("bddl_files"), "libero_spatial", bddl_file)

print(f"  Task: {task.name}")
print(f"  Instruction: {task.language}")
print(f"  BDDL: {bddl_file}")

env_args = {
    "bddl_file_name": bddl_file,
    "camera_heights": 512,
    "camera_widths": 512,
    "has_renderer": False,
    "has_offscreen_renderer": True,
    "render_camera": "agentview",
    "reward_shaping": True,
}

env = OffScreenRenderEnv(**env_args)
env.reset()
print(f"  Robot dof: {env.robots[0].dof}")
print(f"  Env ready!")

# === 2. 远程 VLA 推理 + 生成 GIF ===
print("\n[VLA] Running inference loop, capturing GIF...\n")

setup_tunnel()

import imageio
obs = env.reset()
frames = []
total_steps = 100

for step in range(total_steps):
    # 截图
    img_arr = obs["agentview_image"]
    if img_arr.shape[-1] == 3:
        frame = img_arr[::-1, :, ::-1]  # BGR→RGB
    else:
        frame = img_arr
    frames.append(frame)

    # 编码发送
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # 远程推理
    try:
        resp = requests.post(SERVER_URL,
                           json={"image": img_b64, "instruction": task.language},
                           timeout=15)
        if resp.status_code == 200:
            action = resp.json()["action"]
            action = np.array(action[:7], dtype=float) * 2.0  # 放大动作
        else:
            action = np.zeros(7)
    except Exception as e:
        action = np.zeros(7)
        if step == 0:
            print(f"  VLA connection failed: {e}")

    obs, reward, done, info = env.step(action)

    if step % 20 == 0:
        print(f"  [{step:3d}/{total_steps}] action=({action[0]:+.4f},{action[1]:+.4f},{action[2]:+.4f}...) reward={reward:+.3f}")

env.close()

gif_path = "/media/li/新加卷/isaacsim/workspace/libero_vla.gif"
imageio.mimsave(gif_path, frames, fps=15)
print(f"\n  GIF saved: {gif_path} ({len(frames)} frames)")
print(f"  Open: xdg-open {gif_path}")
