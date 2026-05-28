# VLA Libero 执行计划

Version: execution-v1 | 2026-05-28

## 1. 当前目标

在 Libero/MuJoCo 中完成 OpenVLA-7B 和 PI0-FAST 的可复现实验闭环：

1. 基座模型评估
2. SFT 微调和评估
3. RL 微调和评估
4. 结果对比、文档和 GitHub 整理

主线先跑通 Libero/MuJoCo。Isaac Sim 只作为可选扩展，不阻塞主线进度。

## 2. 当前事实

- 本地工作目录：`/media/li/新加卷/isaacsim/workspace`
- 本地仿真主线：Libero + MuJoCo
- 远程训练/推理：AutoDL RTX 4090D
- OpenVLA server：远程 `6008`，本地隧道 `localhost:8000`
- PI0 server：远程 `6009`，本地隧道 `localhost:6009`
- Isaac Sim 4.5 本地 GUI 渲染有黑屏问题，暂不作为必要路径

## 3. 实验范围

### 必做

| # | 模型 | 阶段 | 环境 | 输出 |
|---|------|------|------|------|
| 1 | OpenVLA-7B | 基座 | Libero/MuJoCo | success rate、reward、latency |
| 2 | PI0-FAST | 基座 | Libero/MuJoCo | success rate、reward、latency、action 格式诊断 |
| 3 | PI0-FAST | SFT | Libero/MuJoCo | 微调 checkpoint、评估结果 |
| 4 | PI0-FAST | RL/PPO | Libero/MuJoCo | RL checkpoint、训练曲线、评估结果 |

### 可选扩展

| 扩展 | 触发条件 | 说明 |
|------|----------|------|
| Isaac Sim 复现实验 | Libero 主线完成后还有时间 | 只做补充验证和展示，不阻塞 SFT/RL |
| Sim-to-Sim 对比 | Isaac Sim 可用后 | 对比 MuJoCo 与 Isaac Sim 的成功率、动作分布和失败模式 |
| OpenVLA RL | PI0 RL 不可行时 | 复用 PPO 环境和评估管道作为备选 |

## 4. 近期优先级

### P0：整理执行基线

- 确认 OpenVLA 基座评估脚本输出稳定落盘
- 统一结果文件命名和字段
- 记录每次评估的模型、checkpoint、任务、episode 数、端口、commit id

### P1：修复 PI0 Libero 接入

- 明确 PI0 输出动作语义：delta joint、absolute joint 或其他格式
- 用真实 Libero robot state 替代零 state
- 决定控制器路径：
  - 优先：PI0 输出转 Libero `JOINT_POSITION`
  - 备选：PI0 输出映射到 `OSC_POSE`
- 跑 1 个任务、3-5 个 episode 的冒烟测试

### P2：SFT

- 下载并整理 Libero spatial 演示数据
- 先用小样本跑通训练和加载
- 再扩大到完整 spatial 数据
- 评估 SFT checkpoint，与基座做同任务对比

### P3：RL/PPO

- 先写环境 rollout、buffer、reward 记录和 checkpoint 保存
- 4090D 上跑短训练，确认 loss、reward、显存和速度
- 再决定是否租更大卡跑正式训练

### P4：文档和展示

- README 写可复现实验流程
- results 中保留 JSON/CSV 指标
- assets 只放压缩后的必要 GIF/MP4，大文件不入 git
- 对外项目摘要等结果稳定后再写，不提前包装

## 5. 工程约束

- 敏感信息不进 git：SSH 密码、token、`.env`、本地 AI 上下文
- 端口统一：
  - OpenVLA：remote `6008` -> local `8000`
  - PI0：remote `6009` -> local `6009`
- 所有新增脚本默认支持 CLI 参数，不把远程地址、密码、checkpoint 写死
- 评估脚本必须能在无 GUI 的情况下保存结果
- 长任务必须写日志，避免只看终端输出
- 每个阶段完成后提交一次小 commit

## 6. 预算策略

| 阶段 | 推荐资源 | 预估 |
|------|----------|------|
| 基座评估 | 4090D | 1-4h |
| SFT 调试 | 4090D | 1-2h |
| SFT 正式 | 4090D 或 32G 卡 | 1-3h |
| RL 调试 | 4090D | 2-5h |
| RL 正式 | VGPU 48G 或 H800 | 10-20h |

原则：先在 4090D 上跑通代码和数据闭环，再租更贵的卡跑长任务。

## 7. 风险和处理

| 风险 | 处理 |
|------|------|
| PI0 action 格式继续不匹配 | 先做状态/action 日志和单步 replay，再决定 controller |
| PI0 SFT 代码成本过高 | 先完成 OpenVLA/PI0 基座对比，SFT 缩小范围 |
| PPO 不收敛 | 保留 rollout、reward 曲线和失败分析；必要时切 OpenVLA RL |
| Isaac Sim 不可用 | 不阻塞主线，只在文档里标记为可选扩展和已知限制 |
| 远程实例重置 | 本地保留脚本、环境说明和 checkpoint 清单 |

## 8. 完成标准

主线完成至少需要：

- OpenVLA 基座 Libero/MuJoCo 评估结果
- PI0 基座 Libero/MuJoCo 评估结果
- PI0 SFT checkpoint 和评估结果
- 一个 RL 调试结果或正式结果
- README 中可复现运行命令
- 结果表能解释成功率、reward、latency 和主要失败模式
