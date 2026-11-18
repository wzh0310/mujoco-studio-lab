#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练入口 wrapper：先注册自定义 push_cube 环境，再进入 LeRobot 官方训练。

为什么需要 wrapper：
  lerobot 0.7.0 的 `register_third_party_plugins()` 只扫描已安装的
  lerobot_env_* / lerobot_policy_* 等 pip 包；我们的环境在
  mujoco-studio-lab/envs/lerobot_push_cube_env.py，不是独立安装的包，
  因此必须在配置解析（draccus 遇到 env.type=push_cube）之前显式 import,
  让 @EnvConfig.register_subclass("push_cube") 完成注册。

用法（在 lerobot conda 环境）：
  python scripts/train_act_push_cube.py --config_path=configs/act_push_cube.yaml
命令行参数原样透传给 lerobot.scripts.lerobot_train.main()（draccus 解析 sys.argv）。
"""
import os
import sys

# 把 mujoco-studio-lab 仓库根加入 sys.path，使 envs/lerobot_push_cube_env 可导入
STUDIO_LAB = os.environ.get(
    "STUDIO_LAB_ROOT", "/data/wzh/robot/mujoco-studio-lab"
)
if STUDIO_LAB not in sys.path:
    sys.path.insert(0, STUDIO_LAB)

# 关键：先 import 触发 @EnvConfig.register_subclass("push_cube") 注册
import envs.lerobot_push_cube_env  # noqa: E402,F401

# 现在进入官方训练入口
from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
