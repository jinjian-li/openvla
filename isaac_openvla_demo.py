"""
Isaac Sim + OpenVLA Joint Demo — all imports before Isaac Sim
"""
import os, json, numpy as np, torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/my_nvidia_icd.json"

print("=" * 60)
print("Isaac Sim + OpenVLA Joint Demo")
print("=" * 60)

# ===== 1. Load everything BEFORE Isaac Sim =====
print("\n[1/3] Loading OpenVLA + data...")
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from PIL import Image

# Model
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", quantization_config=bnb,
    low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto",
)
print("  OpenVLA loaded!")

# Data — grab sample before Isaac Sim's broken PIL loads
ds = load_dataset("Qu3tzal/bridgev2_sample", split="train", streaming=True)
sample = None
for s in ds:
    if s.get("frame_index", 0) > 5: sample = s; break
if sample is None:
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    instruction = "pick up the red block"
else:
    test_img = sample["observation.images.image_0"]  # PIL Image, already decoded
    instruction = sample["language_instruction"]
print(f"  Data ready: {test_img.size if hasattr(test_img, 'size') else 'numpy'}")

# ===== 2. Isaac Sim =====
print("\n[2/3] Starting Isaac Sim...")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})
print("  Isaac Sim started!")

import omni.usd, omni.timeline
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()
stage.DefinePrim("/World/Ground", "Cube")
g = UsdGeom.Cube(stage.GetPrimAtPath("/World/Ground"))
g.AddScaleOp().Set(Gf.Vec3f(3, 3, 0.02))
g.AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.02))

stage.DefinePrim("/World/Gripper", "Sphere")
grip = UsdGeom.Sphere(stage.GetPrimAtPath("/World/Gripper"))
grip.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.3))

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(30): app.update()
print("  Scene ready!")

# ===== 3. Inference =====
print("\n[3/3] Inference loop...")
test_instructions = [
    "move the arm to the left",
    "move the arm to the right",
    "pick up the object",
    "place the object down",
]

results = []
for inst in test_instructions:
    prompt = f"In: What action should the robot take to {inst.lower()}?\nOut:"
    inputs = processor(prompt, test_img).to("cuda", dtype=torch.float16)
    with torch.no_grad():
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig")
    action = np.squeeze(np.array(action))

    dx, dy, dz = float(action[0])*0.1, float(action[1])*0.1, float(action[2])*0.05
    gp = stage.GetPrimAtPath("/World/Gripper")
    if gp.IsValid():
        gp.GetAttribute("xformOp:translate").Set(Gf.Vec3f(dx, dy, 0.3+dz))
    for _ in range(30): app.update()

    r = {"instruction": inst, "action": [f"{a:.4f}" for a in action]}
    results.append(r)
    print(f"  '{inst}' → Δx={action[0]:.4f} Δy={action[1]:.4f} Δz={action[2]:.4f} grip={action[6]:.4f}")

with open("/root/autodl-tmp/demo_results.json", "w") as f:
    json.dump(results, f, indent=2)

timeline.stop(); app.close()
print(f"\n✅ Demo complete! results: demo_results.json")
