#!/usr/bin/env bash
# 启动指定 MJCF 的 MuJoCo Studio WebViewer（无显示器容器 -> 浏览器串流）
# 用法:
#   ./scripts/start_webviewer.sh                      # 默认: 场景骨架 push_cube.xml, 端口 8082
#   ./scripts/start_webviewer.sh model.xml 8081 simulate.py   # 第一个 demo
set -e
MODEL="${1:-assets/scenes/push_cube.xml}"
PORT="${2:-8082}"
PLUGIN="${3:-}"
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mujoco
export MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless
cd "$(dirname "$0")/.."
ARGS="--model=$MODEL --gfx=web --port=$PORT --hot_reload=all"
if [ -n "$PLUGIN" ]; then ARGS="$ARGS --plugins=$PLUGIN"; fi
echo "Starting: mujoco-studio $ARGS"
exec mujoco-studio $ARGS
