# 全量数据训练 + 标准化评估体系

本轮完成「全量数据正式训练」与「评估体系搭建 + 基线量化指标输出」。所有产出在容器
`wzh-ai` 内，涉及 `mujoco-studio-lab`（数据/环境/评估）与 `embodied-policy-zoo`（训练配置）两仓。

## 环境：lerobot 环境 torch 修复（GPU 解锁）

容器内 RTX A6000（驱动 550.107.02 / CUDA 12.4）原本不可用：`lerobot` 环境是
torch 2.15.0+cu132，与驱动不匹配，`torch.cuda.is_available()=False`。
已在 `lerobot` 环境重装 **torch 2.6.0+cu124 / torchvision 0.21.0+cu124**（cu124
官方最高版本，与驱动匹配），CUDA 实测可用。LeRobot 0.7.0 元数据声明要求 torch>=2.7，
但所有核心功能（训练入口 / factory / 数据集加载 / 环境注册）在 2.6.0 下验证正常；
已将 dist-info 元数据下限对齐到 2.6.0 消除 `pip check` 冲突（仅元数据，不改代码）。
训练速度从 CPU 的 0.1 step/s 提升到 **~28 step/s（约 280×）**。

## 数据扩充 + 预处理固化 + 正式训练

### 1. 数据扩充与标准化
- `scripts/collect_demos_v2.py`：在立方体初始位置（±0.03m）之外，新增机械臂 mocap
  起点随机偏移（±0.008m），轨迹规划从真实起点出发，保证观测/动作对齐。
- 从 120 条候选中按「成功优先、距离次之」筛选 **40 条全部成功**的演示
  （早期版本混入 12 条失败轨迹，已废弃）；`data/raw_demos_v2.npz`。
- 导出为 LeRobot v3 数据集 `data/push_cube_demos`：**40 episodes / 1524 frames / fps50**，
  `meta/info.json` 8:2 划分 `train:0:32 / val:32:40`，`stats.json` 按新数据重算，
  `verify` 全部通过（shape / dtype / 时间戳 / 各回合可解码）。
- `scripts/preprocessing.py`：把归一化规则固化为单一真源——图像 uint8→[0,1]→[-1,1]、
  动作 min-max→[-1,1]（按 action_space 边界），并提供与 LeRobot ACT 一致的 z-score
  接口；自检通过。

### 2. 正式训练（GPU）
- 配置 `configs/act_full_train.yaml`：全量 40 episodes、`device=cuda`、`steps=4000`
  （≈50 epoch）、`batch_size=16`、每 5 epoch（`save_freq=800`）存 checkpoint、
  `eval_split=0.2` + `eval_steps=200` 监控验证损失。
- **bf16 坑**：`accelerator.mixed_precision=bf16` 会让 LeRobot 内置 eval rollout 的
  `action.to("cpu").numpy()` 抛 `BFloat16` 错误并中断训练；改用 fp32（A6000 TF32
  加速，YAML 中 `"no"` 必须加引号，否则被解析成 bool）。
- 训练 **4000/4000 步完整结束，零报错**，用时 4 分 11 秒。验证损失单调下降：
  **0.281 → 0.144**（step200→4000），后期趋平，无过拟合。
- checkpoint：`outputs/act_push_cube_full/checkpoints/{000800,001600,002400,003200,004000,last}`。

## 评估体系 + 基线指标 + 演示视频

### 1. 评估脚本 `scripts/eval_act.py`
- 固定 **3 组种子 × 10 回合 = 30 次**（离屏 `MUJOCO_GL=egl`）。
- 每回合注入立方体初始位置随机偏移（与训练数据同分布），保证 30 次独立。
- 指标：**任务成功率**（立方体进入目标半径 3cm）、**平均完成步数**（成功回合）、
  **平均距离误差**（回合末立方体到目标距离）。
- 关键坑：postprocessor（unnormalizer）输出**已是 raw 动作**，不可再做
  `a*std+mean`，否则 dy/dz 被压回均值导致推偏；修正后成功率从 0% 恢复到正常水平。

### 2. 基线量化指标（ckpt 004000）
| 指标 | 数值 |
|---|---|
| 测试回合 | 30（3 seeds × 10） |
| 任务成功率 | **30.0%（9/30）** |
| 平均完成步数（成功） | 36.8 |
| 平均距离误差 | 0.134 m（成功回合 0.023 m） |
| 平均总奖励 | +2.91 |
| 按种子 | seed100 60% / seed200 30% / seed300 0% |

**低于 60% 目标的排查结论**：已将专家数据从「混入失败轨迹」提升到 40/40 全成功，
平均距离误差由 0.186m 降至 0.134m（明显改善），但成功率仍为 30%。剩余瓶颈主要是
评估环境机械臂从固定 home 起步（专家数据起点带随机偏移）、ACT 开环 chunk 执行的
累积误差，以及单一顶视相遮挡。进一步提升需增大数据多样性 / 调 chunk_size 与闭环
频率 / 多相机观测，属后续优化项。当前指标作为**可复现基线对比点**。

### 3. 演示视频
- `assets/eval_success_*.mp4`：单视角（top_cam）成功回合，50fps。
- `assets/eval_dual_view.mp4`：**双视角拼合**（左 global_cam 第三人称 + 右 top_cam
  第一人称），34 帧 / (224, 448, 3)，seed200 成功回合（dist=0.027m）。
  由 `scripts/record_dual_view.py` 生成。

## 复现命令
```bash
# lerobot 环境
python scripts/collect_demos_v2.py gen --episodes 120 --select 40 --seed 2026   # mujoco 环境
python scripts/collect_demos_v2.py export --raw data/raw_demos_v2.npz --out data/push_cube_demos
python scripts/prepare_dataset.py
python scripts/train_act_push_cube.py --config_path=configs/act_full_train.yaml
python scripts/eval_act.py --checkpoint <ckpt>/004000 --seeds 100 200 300 --episodes-per-seed 10 \
    --record-success 2 --out data/eval_act_metrics.json
python scripts/record_dual_view.py --checkpoint <ckpt>/004000 --out assets/eval_dual_view.mp4
```
