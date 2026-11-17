# 场景骨架：Franka Panda + 桌面（push_cube）

这是仓库的第二个 demo：**导入现成机械臂模型，搭好桌面 + 地面基础场景，并验证可视化正常**。
第一个 demo（WebViewer 串流 + 热重载）见 [README](../README.md) 与 [docs/env.md](env.md)。

## 模型来源

机械臂使用官方 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 的
`franka_emika_panda`（不自行建模），完整拷贝到 `assets/franka/`：
- `panda.xml`：9 自由度 Panda + 夹爪，含碰撞/视觉 mesh 与材质；
- `assets/*.stl` / `*.obj`：67 个 mesh 资源；
- 附带官方 `README.md` / `LICENSE` / `CHANGELOG.md`。

> 适配说明：`panda.xml` 的 `<compiler meshdir>` 由官方的 `assets` 改为 `../franka/assets`。
> 这样无论从 `assets/franka/`（模型自身目录）还是从 `assets/scenes/`（主场景目录）加载，
> mesh 路径都指向同一个 `assets/franka/assets/`，避免跨目录 `<include>` 时的路径重复拼接。

## 场景构成（`assets/scenes/push_cube.xml`）

通过 `<include file="../franka/panda.xml"/>` 引入机械臂，主场景只负责环境：

| 元素 | 配置 |
| --- | --- |
| 地面 | `plane`，棋盘格材质 `groundplane` |
| 桌面 | 0.8 m × 0.6 m，顶面高度 **0.75 m**，4 条桌腿；中心 x=0.85（机械臂基座在桌面后方） |
| 待推方块 | 橙色 box，静止放在桌面上（供后续 push 任务） |
| 灯光 | 1 盏 directional 主灯 + 1 盏补光 + headlight |
| 相机 1 `top_cam` | 主俯视相机，正对桌面，用于采集 RGB 观测 |
| 相机 2 `global_cam` | 全局斜侧视角，可看到完整机械臂 + 桌面 |
| 关键帧 `hover` | 末端（夹爪尖）悬停在桌面上方 **0.105 m**；用数值 IK 求得，已验证无穿模 |

机械臂底座固定在桌面后方地面，末端前伸到桌面后部上方；经 MuJoCo 接触检测确认
`pens=0`（无穿模）。

## 怎么跑

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mujoco
export MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless
cd /data/wzh/robot/mujoco-studio-lab
mujoco-studio --model=assets/scenes/push_cube.xml --gfx=web --port=8082 --hot_reload=all
```

浏览器打开提示地址（`http://localhost:8082`，容器端口需 SSH 隧道转发，见 [env.md](env.md)）。
在左侧 Simulation 面板可 Pause / Run、拖时间轴、切换 Keyframe（`hover`）；
鼠标左键拖动改变视角。

## 验证结果

![全局视角](assets/scene_default.png)

*默认全局视角：机械臂位于桌面后方，末端悬停在桌面上方，橙色方块在桌面上，无穿模。*

![拖动视角后](assets/scene_dragged.png)

*鼠标左键拖动后的视角，画面正常、无报错。*

双相机离屏渲染（`MUJOCO_GL=egl`）：

| `top_cam`（俯视，RGB 观测） | `global_cam`（全局斜侧） |
| --- | --- |
| ![top_cam](assets/top_cam.png) | ![global_cam](assets/global_cam.png) |

验证项：机械臂与桌面无穿模（`pens=0`）、末端悬停高度 0.105 m、两个相机画面正常、
拖动视角 / 暂停播放无报错（浏览器 `Errors: 0`）。

## 当日产出

- `assets/franka/`：官方 Panda 模型（含 mesh/材质）
- `assets/scenes/push_cube.xml`：基础场景骨架（地面 + 桌面 + 双相机 + hover 关键帧）
- `assets/{scene_default,scene_dragged,top_cam,global_cam}.png`：可视化验证截图
- `docs/scene.md`：本说明
