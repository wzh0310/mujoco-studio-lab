# 环境与复现

本仓库的 demo 在一台 Docker 容器（`wzh-ai`）里开发，通过 SSH 隧道把
MuJoCo WebViewer 串流到本机浏览器。

## 运行环境

| 项目 | 值 |
| --- | --- |
| 宿主 | `172.16.45.13`（SSH 端口 2026，用户 `lxt`） |
| 容器 | `wzh-ai`（Ubuntu 22.04，镜像 `wzh-ai:ssh-ready`） |
| conda 环境 | `/opt/miniconda3/envs/mujoco`（Python 3.12） |
| MuJoCo | 4.0.1（默认元包，含 `mujoco-core` + `mujoco-studio`） |
| 其他包 | gymnasium 1.4.0、stable-baselines3 2.9.2、wandb 0.31.0 |
| GPU | NVIDIA RTX A6000 48 GB，驱动 550.107.02 |
| 渲染 | 无头 EGL：`MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless` |

## 环境搭建

```bash
conda create -n mujoco python=3.12 -y
conda activate mujoco
pip uninstall -y mujoco          # 从 3.x 升级时必须先卸载
pip install mujoco               # 4.0 会同时装 mujoco-core + mujoco-studio
pip install gymnasium stable-baselines3 wandb
python -c "import mujoco; print(mujoco.__version__)"   # 4.0.1
```

## 启动 WebViewer（容器内）

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mujoco
export MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless
mujoco-studio --model=model.xml --plugins=simulate.py \
    --gfx=web --port=8080 --hot_reload=all
```

`--gfx=web` 的关键含义：物理与 UI 分离在两端，通过非阻塞消息通道同步。
Filament 负责 PBR 渲染，Dear ImGui 负责界面；无 GPU / 无显示器的机器跑物理，
浏览器只做串流显示。一个浏览器是 controller（可驱动仿真），其余为 spectator。
`--hot_reload=all` 让 MJCF / 插件文件改动即时生效。

## 本机访问（SSH 隧道）

容器 8080 已映射到宿主 `172.16.45.13:8080`。在本机建隧道后打开浏览器：

```bash
ssh -N -L 18080:172.16.45.13:8080 dev13
# 浏览器打开 http://localhost:18080
```

MuJoCo WebViewer 要求安全上下文（HTTPS 或 localhost），因此不要直接用
`http://172.16.45.13:8080`，要走 localhost 隧道。

## 验证热重载

1. 浏览器打开 `http://localhost:18080`，看到橙色小球上下运动；
2. 拖动鼠标左键改变视角，画面实时响应；
3. 编辑 `model.xml` 里 sphere 的 `rgba`（第三个分量 `0.3` → `1.0`），保存；
4. 浏览器自动重载，小球变蓝 —— 证明 MJCF 热重载生效（无需重启仿真）。
