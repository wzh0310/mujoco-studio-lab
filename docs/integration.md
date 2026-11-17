

---

## Demo7: 全链路压测、Bug 修复与代码整理

目标：跑通「环境启动 → 交互 → 数据采集」全链路，修复 bug、优化体验、统一目录结构。
全部在 13 服务器 `wzh-ai` 容器内完成（可视化用 `mujoco` conda 环境，LeRobot 数据用 `lerobot` 环境）。

### 1. 全链路压测（`scripts/stress_test.py`）

一条命令复现：`python scripts/stress_test.py`（mujoco 环境 + EGL）。报告 JSON 末尾 `PASS` 为真即通过。

- **A. 连续 10 回合专家生成**：400 帧，**0 崩溃 / 0 NaN / 0 黑帧**；8/10 回合终距 <0.03m 成功（2 条因初始偏移落在臂工作空间边界，属策略精度问题，非崩溃）。
- **B. 观测-动作时间对齐**：`len(images)==len(actions)==len(states)==sum(episode_lengths)==183`；fps=50 与物理步长 0.02s 一致；回合内立方体状态最大单步跳变 0.027m（连续、无错位帧）；图像帧间差均值 2.2（非冻结/黑屏流）。
- **C. 重置稳定性**：20 次随机 seed 重置，机械臂关节 + 末端 + 立方体位置最大偏差 **2.2e-16**（机器精度，完全可复现）。

### 2. 发现并修复的 Bug

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | **首帧数据错位 0.021m** | `collect_demos.py` 注入立方体初始偏移后未重新取 obs，首帧 image/state 仍是偏移前的立方体位置 | 注入偏移 + `mj_forward` 后调用 `obs = env._get_obs()` 并同步 `_prev_dist`；已重新生成数据集（5 episodes / 183 frames，VERIFY OK） |
| 2 | **穿模检测全报 0 的假象** | MuJoCo 4.0 的 `MjContact` 没有 `body1`/`body2` 属性，须用 `model.geom_bodyid[c.geom1]` 解析 body id | 封装为 `utils.validation.contact_penetrations()`，压测得到真实穿透深度 |

### 3. 穿模调查与结论（不改物理参数的依据）

用正确 API 实测：接触主要发生在推动阶段的 `right_finger`（最深 3.6cm）、`hand`（1.4cm）、`link5/link6`（2.4cm），是 mocap+weld **软约束**（`solref=0.01`）在末端跟随时的正常接触穿透。

- 目视确认：最深穿模帧的 224×224 top_cam 观测图**看不出任何穿模**，画面正常，不影响训练观测。
- 调参实验（均未采用，记录在案）：
  - 硬化桌面接触 `solref=0.002`：穿模降到 2.6cm，但推物成功率 0/3（立方体滑动动力学被破坏）。
  - 调硬 weld `solref≤0.005`：穿模反而升到 6.5cm、成功率 0/4，甚至出现 `QACC NaN` 不稳定。
  - 抬高推物高度 0.88/0.92：成功率下降或穿模无明显改善。
- **结论**：baseline（`weld solref=0.01`, `push_z=0.86`）是穿模/成功率/稳定性三者的最优点；接触穿透是软约束方案固有特性且观测不可见，保持物理参数不动。这是正确的工程权衡，而非遗漏。

### 4. 体验优化

- 动作边界裁剪已就位：`step()` 里动作 clip 到 `±0.02m`，mocap 目标再 clip 到工作空间盒 `x∈[0.10,0.85], y∈[-0.25,0.25], z∈[0.82,1.15]`，避免超界与无效探索。
- 可视化体验：场景已有 skybox/棋盘地面/木质桌面/半透明目标盘 + 主平行光 + 补光（Demo2/3 完成），观测图清晰美观，可直接用于演示录像。

### 5. 代码结构化整理

统一目录（新增 `utils/`，压测脚本入 `scripts/`）：

```
mujoco-studio-lab/
├── assets/                    # 模型、场景 XML、材质、验证截图
│   ├── franka/                # 官方 Panda 模型 + panda_mocap.xml（无 actuator）
│   ├── scenes/push_cube.xml   # 任务场景（mocap+weld + 立方体 + 目标区 + 双相机）
│   └── *.png / demo.*         # 可视化验证与热重载录屏
├── envs/                      # 环境封装 + 配套测试
│   ├── push_cube_env.py       # Gymnasium 环境（观测/动作/奖励/终止/步进）
│   ├── test_env.py            # 环境冒烟测试
│   └── test_reward.py         # 奖励/终止断言测试
├── scripts/                   # 数据采集、压测、录制、启动脚本
│   ├── collect_demos.py       # 专家演示生成 + LeRobot 导出 + 读取验证（三段式）
│   ├── stress_test.py         # 全链路压测（本次新增）
│   ├── record_demo.py         # WebViewer 录屏复现
│   └── start_*.sh             # WebViewer / 推物 demo 一键启动
├── utils/                     # 可复用工具（本次新增）
│   ├── vision.py              # 离屏渲染 / PNG 导出 / 黑帧检测 / 帧差
│   └── validation.py          # 接触穿透检查 / 数据对齐校验 / 重置稳定性
├── docs/                      # env / scene / task 说明
├── data/                      # raw_demos.npz（gitignore）+ push_cube_demos（入库）
├── push_sim.py                # 推物动画 plugin
├── model.xml / simulate.py    # Demo1 热重载最小场景
└── README.md
```

命名与注释规范：模块/函数用 snake_case，常量大写；公共函数均带 docstring 说明用途与返回；`utils` 把压测里易错的 MuJoCo 4.0 接触 API 与对齐校验收敛为单一实现，避免重复踩坑。

### 6. 当日产出

- 稳定可复现的环境与采集工具：`scripts/stress_test.py` 一条命令跑完全链路并给出 PASS/FAIL。
- 修复后的数据集：`data/push_cube_demos/`（5 episodes / 183 frames，首帧对齐正确）。
- 可复用工具库 `utils/`，统一目录结构与文档。
