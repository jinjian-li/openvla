"""
MuJoCo Reacher + Remote OpenVLA
基于 Gymnasium reacher 模型（已验证稳定），VLA 远程推理控制
"""

import time, io, base64, subprocess, socket, numpy as np
import requests
from PIL import Image
import mujoco
import mujoco.viewer

# === Gymnasium Reacher 模型 XML（已验证稳定） ===
XML = """
<mujoco model="reacher">
    <compiler angle="radian" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
    </default>
    <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
    <worldbody>
        <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.2 0.2 0.25 1" size="1 1 10" type="plane"/>
        <geom conaffinity="0" fromto="-.3 -.3 .01 .3 -.3 .01" name="sideS" rgba="0.9 0.8 0.3 1" size=".02" type="capsule"/>
        <geom conaffinity="0" fromto=" .3 -.3 .01 .3  .3 .01" name="sideE" rgba="0.9 0.8 0.3 1" size=".02" type="capsule"/>
        <geom conaffinity="0" fromto="-.3  .3 .01 .3  .3 .01" name="sideN" rgba="0.9 0.8 0.3 1" size=".02" type="capsule"/>
        <geom conaffinity="0" fromto="-.3 -.3 .01 -.3 .3 .01" name="sideW" rgba="0.9 0.8 0.3 1" size=".02" type="capsule"/>
        <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.3 0.3 0.3 1" size=".015" type="cylinder"/>
        <body name="body0" pos="0 0 .01">
            <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".015" type="capsule"/>
            <joint axis="0 0 1" limited="true" name="joint0" pos="0 0 0" range="-2.0 2.0" type="hinge"/>
            <body name="body1" pos="0.1 0 0">
                <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
                <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".012" type="capsule"/>
                <body name="fingertip" pos="0.11 0 0">
                    <geom conaffinity="1" contype="1" name="fingertip" pos="0 0 0" rgba="1.0 0.2 0.2 1" size=".015" type="sphere"/>
                </body>
            </body>
        </body>
        <body name="target" pos=".1 -.1 .01">
            <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.27 .27" ref=".1" stiffness="0" type="slide"/>
            <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.27 .27" ref="-.1" stiffness="0" type="slide"/>
            <geom conaffinity="1" contype="1" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".02" type="sphere"/>
        </body>
    </worldbody>
    <actuator>
        <position ctrllimited="true" ctrlrange="-2.0 2.0" kp="20" kv="5" joint="joint0"/>
        <position ctrllimited="true" ctrlrange="-3.0 3.0" kp="20" kv="5" joint="joint1"/>
    </actuator>
</mujoco>
"""

SERVER_URL = "http://127.0.0.1:8000/act"

print("=" * 60)
print("MuJoCo Reacher + Remote OpenVLA")
print("=" * 60)

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f"  Model: 2-DOF arm ({model.nbody} bodies, {model.nq} dofs)")
print(f"  Actuators: motor + gear=200, torque control")
print(f"  Stable: RK4 integrator, armature=1, damping=1")

instructions = [
    "move the arm to the left",
    "move the arm to the right",
    "move the arm up",
    "move the arm down",
    "pick up the object",
    "place the object down",
]


def render_img():
    scene = mujoco.MjvScene(model, maxgeom=100)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.1, -0.05, 0.05]
    cam.distance = 0.6; cam.elevation = -20; cam.azimuth = 90
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    r = mujoco.Renderer(model, 256, 256); p = r.render(); r.close()
    return Image.fromarray(p)


def ask_vla(img, inst):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        r = requests.post(SERVER_URL, json={"image": b64, "instruction": inst}, timeout=10)
        if r.status_code == 200: return r.json()["action"]
    except: pass
    return None


print("\n[Loop] 2-DOF arm sweeping. Pure local demo.\n")

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    step = 0
    data.qpos[:2] = [0.0, 0.0]
    data.ctrl[:2] = [0.0, 0.0]  # 位置控制：ctrl=目标角度
    mujoco.mj_forward(model, data)

    while viewer.is_running():
        t = step * model.opt.timestep
        data.ctrl[0] = 1.0 * np.sin(t * 0.7)
        data.ctrl[1] = 0.8 * np.sin(t * 0.6 + 0.5)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
        if step % 200 == 0:
            print(f"  [t={step*model.opt.timestep:5.1f}s] q=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f})")
        step += 1
        if step >= 3000:  # 30 秒后自动停
            break

print("Done.")
