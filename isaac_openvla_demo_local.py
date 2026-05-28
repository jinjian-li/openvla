#!/usr/bin/env python3
"""
Isaac Sim + OpenVLA Joint Demo — local GUI mode for RTX 4060 dual‑GPU laptop.
Remote: /root/autodl-tmp/isaac_openvla_demo.py
"""

import os, json, sys, numpy as np, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/media/li/新加卷/isaacsim/hf_cache/huggingface"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

# Dual-GPU: force NVIDIA dGPU
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(SCRIPT_DIR, "demo_results.json")

import torch

print("=" * 60)
print("Isaac Sim + OpenVLA Joint Demo  (local GUI)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# ============ 1. Load OpenVLA BEFORE Isaac Sim ============
print("\n[1/3] Loading OpenVLA‑7B (4‑bit) + data...")
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b", trust_remote_code=True
)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=bnb,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
    attn_implementation="eager",
)
vram = torch.cuda.mem_get_info()
print(f"  OpenVLA loaded! VRAM free: {(vram[0]-vram[1])/1e9:.1f} / {vram[0]/1e9:.1f} GB")

# Grab a sample image before Isaac Sim
from PIL import Image as PILImage

ds = load_dataset("Qu3tzal/bridgev2_sample", split="train", streaming=True)
sample = None
for s in ds:
    if s.get("frame_index", 0) > 5:
        sample = s
        break
if sample is None:
    test_img = PILImage.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )
    instruction = "pick up the red block"
else:
    test_img = sample["observation.images.image_0"]
    instruction = sample["language_instruction"]
print(f"  Data ready: {instruction}")

# ============ 2. Isaac Sim GUI ============
print("\n[2/3] Starting Isaac Sim GUI ...")

from isaacsim import SimulationApp

# Patch _wait_for_viewport to add timeout (RTX fails on 535 Vulkan, viewport never ready)
_original_wait = SimulationApp._wait_for_viewport
def _patched_wait(self):
    try:
        from omni.kit.viewport.utility import get_active_viewport
        viewport_api = get_active_viewport()
        frame = 0
        while viewport_api.frame_info.get("viewport_handle", None) is None:
            self._app.update()
            frame += 1
            if frame > 120:
                print("  Viewport warmup timeout, continuing...")
                break
    except Exception:
        pass
    for _ in range(10):
        self._app.update()
SimulationApp._wait_for_viewport = _patched_wait

app = SimulationApp({"headless": False, "width": 1280, "height": 720})
print("  Isaac Sim GUI started!")

import omni.usd, omni.timeline
from pxr import UsdGeom, UsdPhysics, Gf, Sdf

stage = omni.usd.get_context().get_stage()

# Ground plane
ground_path = Sdf.Path("/World/Ground")
UsdGeom.Xform.Define(stage, ground_path)
ground = UsdGeom.Cube.Define(stage, ground_path.AppendChild("Cube"))
ground.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.02))
ground.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.02))

# A simple target cube
cube_path = Sdf.Path("/World/TargetCube")
UsdGeom.Xform.Define(stage, cube_path)
target = UsdGeom.Cube.Define(stage, cube_path.AppendChild("Geom"))
target.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, 0.05))
target.AddTranslateOp().Set(Gf.Vec3f(0.3, 0.1, 0.05))
target.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.2, 0.2)])

# Gripper / end-effector sphere (visualizes the action)
gripper_path = Sdf.Path("/World/Gripper")
UsdGeom.Xform.Define(stage, gripper_path)
grip = UsdGeom.Sphere.Define(stage, gripper_path.AppendChild("Sphere"))
grip.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.3))
grip.AddScaleOp().Set(Gf.Vec3f(0.04, 0.04, 0.04))
grip.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.6, 1.0)])

# Text label for current instruction (shown on a prim)
label_path = Sdf.Path("/World/InstructionLabel")
UsdGeom.Xform.Define(stage, label_path)

timeline = omni.timeline.get_timeline_interface()
timeline.play()

# Let the scene settle
print("  Warming up...")
for _ in range(60):
    app.update()
print("  Scene ready!")

# ============ 3. Inference + visual loop ============
print("\n[3/3] Running inference — watch the blue sphere move!")

test_instructions = [
    "move the arm to the left",
    "move the arm to the right",
    "move the arm forward",
    "move the arm backward",
    "pick up the object",
    "place the object down",
]

results = []
for idx, inst in enumerate(test_instructions):
    prompt = f"In: What action should the robot take to {inst.lower()}?\nOut:"
    inputs = processor(prompt, test_img).to("cuda", dtype=torch.float16)

    with torch.no_grad():
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig")
    action = np.squeeze(np.array(action))

    dx = float(action[0]) * 0.1
    dy = float(action[1]) * 0.1
    dz = float(action[2]) * 0.05

    # Animate the gripper sphere to the new position
    gp = stage.GetPrimAtPath("/World/Gripper/Sphere")
    if gp.IsValid():
        attr = gp.GetAttribute("xformOp:translate")
        cur = Gf.Vec3d(attr.Get()) if attr.Get() else Gf.Vec3d(0, 0, 0.3)
        target_pos = Gf.Vec3d(
            cur[0] + dx * 0.5,
            cur[1] + dy * 0.5,
            max(0.05, cur[2] + dz * 0.5),
        )
        # Smooth step
        steps = 40
        for s in range(steps):
            t = (s + 1) / steps
            interp = cur + (target_pos - cur) * t
            attr.Set(interp)
            app.update()

    r = {
        "instruction": inst,
        "action": [f"{v:.4f}" for v in action],
        "delta": [f"{dx:.4f}", f"{dy:.4f}", f"{dz:.4f}"],
    }
    results.append(r)
    status = f"Δx={action[0]:+.4f}  Δy={action[1]:+.4f}  Δz={action[2]:+.4f}  grip={action[6]:.4f}"
    print(f"  [{idx+1}/{len(test_instructions)}] '{inst}' → {status}")

    # Pause so user can see the result
    time.sleep(0.5)

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)

timeline.stop()
app.close()

print(f"\n✅ Demo complete!")
print(f"   Results saved to: {OUTPUT}")
print(f"   Press Ctrl+C if window doesn't close automatically.")
sys.exit(0)
