"""
Isaac Sim VLA 客户端 — CS 架构远程推理 + 本地仿真
远程: AutoDL 4090D 运行 OpenVLA-7B (vla_server.py)
本地: Isaac Sim 4.5 GUI + 535 驱动 (RTX 视口黑屏已知, 仿真管道正常)
"""

import os, io, base64, time, subprocess, sys, numpy as np

# === 环境配置 ===
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

import requests
from PIL import Image

# === 远程服务器 ===
SERVER_URL = "http://127.0.0.1:8000/act"
SSH_PASSWORD = os.getenv("AUTODL_SSH_PASSWORD")
SSH_TUNNEL_CMD = [
    "sshpass", "-p", SSH_PASSWORD or "",
    "ssh", "-p", "54812", "-o", "StrictHostKeyChecking=no",
    "-N", "-L", "8000:127.0.0.1:6008",
    "root@connect.westc.seetacloud.com",
]


def setup_tunnel():
    """确保 SSH 隧道已连接"""
    if not SSH_PASSWORD:
        raise RuntimeError("Missing AUTODL_SSH_PASSWORD environment variable.")
    import socket
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", 8000))
        s.close()
        return True
    except:
        s.close()
        print("  Starting SSH tunnel...")
        subprocess.Popen(SSH_TUNNEL_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        return False


print("=" * 60)
print("Isaac Sim + Remote OpenVLA  (CS Architecture)")
print("=" * 60)

# ===== 1. Isaac Sim GUI =====
print("\n[1/3] Starting Isaac Sim GUI ...")
print("  (black viewport on 535 driver, simulation works correctly)")

from isaacsim import SimulationApp

# RTX fails on 535 → viewport handle never ready → need timeout
_orig = SimulationApp._wait_for_viewport
def _patched(self):
    try:
        from omni.kit.viewport.utility import get_active_viewport
        api = get_active_viewport()
        for _ in range(120):
            if api.frame_info.get("viewport_handle", None) is not None:
                break
            self._app.update()
        else:
            print("  Viewport warmup timeout, continuing...")
    except Exception:
        pass
    for _ in range(10):
        self._app.update()
SimulationApp._wait_for_viewport = _patched

app = SimulationApp({"headless": False, "width": 1280, "height": 720})
print("  Isaac Sim GUI started!")

import omni.usd, omni.timeline
from pxr import UsdGeom, UsdLux, Gf, Sdf

stage = omni.usd.get_context().get_stage()

# 场景: 地面 + 红色球体 + 灯光
ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Ground"))
ground.AddScaleOp().Set(Gf.Vec3f(3, 3, 0.05))
ground.AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.05))
ground.CreateDisplayColorAttr().Set([(0.3, 0.3, 0.3)])

sphere = UsdGeom.Sphere.Define(stage, Sdf.Path("/World/Target"))
sphere.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.5))
sphere.AddScaleOp().Set(Gf.Vec3f(0.15, 0.15, 0.15))
sphere.CreateDisplayColorAttr().Set([(1.0, 0.2, 0.2)])

light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
light.CreateIntensityAttr().Set(500.0)

# 尝试调整相机
try:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    if vp:
        vp.camera_position = (1.5, 1.5, 2.0)
        vp.camera_target = (0.0, 0.0, 0.3)
except:
    pass

print("  Scene ready: ground + red sphere + dome light")

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(60):
    app.update()

# ===== 2. 推理循环 =====
print("\n[2/3] Starting inference loop ...")
setup_tunnel()

instructions = [
    "move the arm to the left",
    "move the arm to the right",
    "move the arm up",
    "move the arm down",
    "pick up the object",
    "place the object down",
]

pos = [0.0, 0.0, 0.5]
for i, inst in enumerate(instructions):
    # 截图 (fallback: 灰色图像)
    try:
        import omni.syntheticdata as sd
        rgb = sd.acquire_synthetic_data("rgb")
        img = Image.fromarray(rgb) if rgb is not None else Image.new("RGB", (256, 256))
    except:
        img = Image.new("RGB", (256, 256))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # 远程推理
    try:
        resp = requests.post(SERVER_URL, json={"image": img_b64, "instruction": inst}, timeout=10)
        if resp.status_code == 200:
            action = resp.json()["action"]
        else:
            print(f"  Server error {resp.status_code}")
            continue
    except Exception as e:
        print(f"  Connection failed: {e}")
        action = [np.random.randn() * 0.01 for _ in range(7)]

    dx = float(action[0]) * 0.1
    dy = float(action[1]) * 0.1
    dz = float(action[2]) * 0.05
    pos[0] += dx
    pos[1] += dy
    pos[2] += dz

    sphere.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(*pos))
    for _ in range(30):
        app.update()

    print(f"  [{i+1}/{len(instructions)}] {inst}")
    print(f"       dx={dx:+.4f} dy={dy:+.4f} dz={dz:+.4f}  pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")

# ===== 3. 结束 =====
print(f"\n[3/3] Demo complete!  Sphere moved through {len(instructions)} positions.")
print("  Window stays open 30s — look at the viewport (Ctrl+C to close)")
timeline.stop()
time.sleep(30)
app.close()
