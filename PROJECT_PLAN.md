# 具身智能 VLA 项目计划书 v3

## 1. 项目定位

**主线：** PI0-FAST 全流程（基座 → SFT微调 → RL微调）→ MuJoCo/Isaac Sim 仿真验证
**对照：** OpenVLA-7B 基座模型 → MuJoCo/Isaac Sim 仿真验证
**亮点：** 双模型对比 + 强化学习微调 + Sim-to-Sim 跨仿真器分析

## 2. 实验矩阵

```
                    PI0-FAST (主线)               OpenVLA-7B (对照)
                   ┌──────┼──────┐              ┌──────┘
                   │      │      │              │
                基座    SFT    RL              基座
                 ✅      ✅     ✅              ✅
                 │      │      │              │
                 └──────┴──────┘              │
                        │                     │
                   MuJoCo 仿真                 │
                   Isaac Sim 仿真              │
                        │                     │
                   Sim-to-Sim                 │
```

**4 组实验：**

| # | 模型 | 训练 | 仿真 |
|---|------|------|------|
| 1 | PI0-FAST | 基座 | MuJoCo + Isaac Sim |
| 2 | PI0-FAST | SFT LoRA | MuJoCo + Isaac Sim |
| 3 | PI0-FAST | RL (PPO) | MuJoCo + Isaac Sim |
| 4 | OpenVLA-7B | 基座 (4-bit) | MuJoCo + Isaac Sim |

## 3. GPU 预算和租卡策略

### 可用 GPU 和价格

| GPU | 显存 | 单价 | 适合什么 |
|-----|------|------|----------|
| 4090/4090D | 24GB | ¥1.88 | 推理、SFT微调、代码调试 |
| VGPU 32G | 32GB | ¥1.68 | 推理、SFT微调 |
| 5090/5090D | 32GB | ¥2.78 | SFT微调、RL调试 |
| VGPU 48G | 48GB | ¥2.88 | RL训练（性价比高） |
| RTX PRO 6000 | 48GB | ¥5.98 | RL训练 |
| H800 | 80GB | ¥8.88 | RL正式训练（最快） |

### 各阶段预算

| 阶段 | 任务 | 推荐卡 | 预计时长 | 费用 |
|------|------|--------|----------|------|
| 1 | PI0 基座评估 | 4090D | 1-2h | ¥2-4 |
| 2 | OpenVLA 基座评估 | 4090D | 1-2h | ¥2-4 |
| 3 | PI0 SFT 微调 | 4090D | 30min | ~¥1 |
| 4 | PI0 SFT 评估 | 4090D | 1-2h | ¥2-4 |
| 5 | **RL 代码调试** | 4090D | 2-4h | ¥4-8 |
| 6 | **RL 正式训练** | H800 或 VGPU48G | 10-20h | ¥90-180 |
| 7 | RL 模型评估 | 4090D | 1-2h | ¥2-4 |
| 8 | Isaac Sim 渲染 | 本地 4060 | - | 免费 |

**总预算估算：¥110-210**

> 策略：代码调试和短任务用 4090D（¥1.88/h），确定没问题后切 H800 跑 RL 正式训练。不租双卡——单 H800 足够。
> 
> Isaac Sim headless 渲染用本地 4060（免费），或者租 4090D 时顺便渲染（4090 有 RT Core）。

## 4. 为什么主推 PI0 而不是 OpenVLA？

- PI0 是 **Flow Matching** 架构，动作连续自然，比 OpenVLA 的离散 token 更适合精细操作
- 远程已有 `pi0_fast_libero` checkpoint（11GB），不是从零开始
- OpenVLA 已经做过基座评估了，做对照就行——重复做全套价值不大
- 两个模型同时做全套确实是重复劳动，省下的时间可以加深 PI0 RL

**PI0 RL 可行性：** PI0 的 Flow Matching 采样过程是可微的，可以作为 stochastic policy 接入 PPO。JAX 版本的 PPO 有现成实现（`purejaxrl`）。

**如果 PI0 RL 不行怎么办：** RL 环境（Libero + reward）是模型无关的，写一次复用。PI0 RL 卡住了，半天就能把 OpenVLA 接上去跑 RL。不浪费时间。

## 5. 技术细节

### 5.1 PI0 推理管道

```
本地 Libero 截图 → base64 → POST /act (远程 4090D)
                              ↓
                    PI0 模型推理 (JAX)
                              ↓
                    返回 7-dim 关节动作
                              ↓
                    Libero env.step()
```

远程 `pi0_server.py` 已在端口 6007。接入方式与 OpenVLA 完全一致（POST 图片+指令→返回动作），本地客户端改一行 URL 即可。

### 5.2 PI0 SFT 微调 (LoRA)

复用远程已有的 `finetune_v2.py` 框架，将模型从 OpenVLA 换成 PI0，数据源换成 Libero。

- 方法：LoRA rank=32
- 数据：Libero spatial 10 任务演示数据
- 时长：~30 分钟（4090D）

### 5.3 PI0 RL 微调 (PPO)

```
循环:
  ┌─ Libero env.reset()
  ├─ PI0 采样动作 a ~ π(·|image)
  ├─ env.step(a) → reward, next_image
  ├─ 存入 buffer: (image, a, reward, next_image)
  ├─ 每 N 步更新:
  │    - PPO clipped loss 更新 action head (LoRA)
  │    - Critic (小型 MLP, 共享 vision feature) 回归 value
  └─ 直到 reward 收敛
```

关键设计：
- 冻结视觉 backbone（不训，省显存）
- 只更新 LoRA + action head
- Critic 是独立小网络（~2M 参数），共享 ViT 特征
- 奖励：Libero 自带 task reward

### 5.4 Sim-to-Sim 对比

```
同一任务 "pick up the bowl"
    ├── MuJoCo (Libero)  → 录 50 轮 action 轨迹
    └── Isaac Sim        → 录 50 轮 action 轨迹
                              ↓
              对比：动作分布差异、成功率差异
```

## 6. 评估指标

| 指标 | 说明 |
|------|------|
| Success Rate | Libero 自动评测 |
| Avg Reward | 每 episode 平均 |
| Action Distribution | 动作均值/方差对比 |
| Inference Time | ms/step |
| Sim-to-Sim MSE | 同一模型在两 sim 上的动作差异 |

**最终对比表（README 核心）：**

| 模型 | 训练 | MuJoCo SR | Latency |
|------|------|-----------|---------|
| OpenVLA-7B | 基座 | XX% | XXms |
| PI0-FAST | 基座 | XX% | XXms |
| PI0-FAST | SFT | XX% | XXms |
| PI0-FAST | RL | XX% | XXms |

## 7. GitHub 仓库

```
jinjian-li/vla-libero-benchmark/
├── README.md                 ← 总览 + 对比表 + GIF
├── docs/
│   ├── setup.md              ← 环境搭建（本地+远程）
│   ├── pipeline.md           ← 推理管道架构
│   ├── training.md           ← SFT + RL 微调方案
│   └── results.md            ← 完整评估报告
├── scripts/
│   ├── libero_eval.py        ← 批量评估
│   ├── libero_pi0_client.py  ← PI0 客户端
│   ├── libero_vla_client.py  ← OpenVLA 客户端
│   ├── libero_ppo_trainer.py ← PPO 训练器
│   └── isaac_vla_client.py   ← Isaac Sim 客户端
├── remote/
│   ├── pi0_server.py         ← PI0 推理服务
│   ├── vla_server.py         ← OpenVLA 推理服务
│   ├── finetune_sft.py       ← SFT LoRA 微调
│   └── finetune_ppo.py       ← PPO 微调
├── assets/demos/             ← GIF/MP4
├── results/                  ← JSON/CSV
└── configs/                  ← 超参数
```

## 8. 简历展示

**项目名称:** VLA 模型对比评测与 RL 微调 — PI0/OpenVLA × MuJoCo/Isaac Sim 双平台验证

**技术栈:** PI0-FAST, OpenVLA-7B, PPO, LoRA, Flow Matching, MuJoCo, Isaac Sim, Libero, JAX, PyTorch

**核心产出 (预期):**
- 构建 PI0/OpenVLA 双模型在 LIBERO 10 项操作任务上的标准化评测管道
- 实现 PPO 强化学习微调 VLA 模型（冻结视觉 backbone + LoRA action head + 独立 Critic）
- 完成 MuJoCo/Isaac Sim 双仿真器对比，分析物理引擎差异对策略的影响
- 产出 4 组模型的完整性能对比表 + 并排演示 GIF

## 9. 时间线

| 阶段 | 内容 | 时长 | 租卡 |
|------|------|------|------|
| 1 | PI0 基座 + OpenVLA 基座评估 | 3-4h | 4090D ¥8 |
| 2 | PI0 SFT 微调 + 评估 | 2-3h | 4090D ¥6 |
| 3 | RL 代码调试 (4090D) | 3-5h | 4090D ¥10 |
| 4 | RL 正式训练 (H800) | 10-20h | H800 ¥180 |
| 5 | 全模型评估 + Sim-to-Sim | 3-4h | 4090D ¥8 |
| 6 | 文档 + 简历 | 与 4/5 并行 | - |

**实际总时长：~1.5-2 周**（阶段 4 纯训练等待，期间并行做阶段 6）

**总预算：~¥220**

## 10. 风险预案

| 风险 | 概率 | 应对 |
|------|------|------|
| PI0 RL 训练不收敛 | 低-中 | RL 环境复用，切换 OpenVLA 接 PPO，半天 |
| H800 租不到 | 低 | VGPU48G (¥2.88/h) 替代，慢 2-3× 但够用 |
| PI0 JAX 版本兼容问题 | 中 | 固定版本，远程已有 working 环境 |

---

Version: v3 | 2026-05-27
