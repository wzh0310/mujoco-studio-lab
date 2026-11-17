#!/usr/bin/env bash
# 启动推物任务 demo：push_cube 场景 + mocap 驱动 plugin（端口 8083）
set -e
DIR="$(dirname "$0")/.."
cd "$DIR"
exec ./scripts/start_webviewer.sh assets/scenes/push_cube.xml 8083 push_sim.py
