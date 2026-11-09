# AirSim 短路径视觉导航演示 Runbook

**目标**：在仿真中识别指定物体（如 `car`），自主规划并接近该物体。**不需要 Orin / 真机**。

**栈**：`wam_vgoal_eval` — YOLO → TargetTracker → π + ImaginationPlanner → AirSim。

**主机**：125（4090 + AirSim，`env_4090.sh`）；`.110` 已退役，勿用。

---

## 0. 前置条件

| 项 | 说明 |
|----|------|
| 代码 | `~/aerial-wam-v2` 与 Mac 同步 |
| venv | `~/sim_verify/.venv`（`env_4090.sh` 自动选用） |
| vgoal 仓库 | `~/Projects/aerial-vgoal-wam`（检测器 / tracker） |
| AirSim | `AERIAL_PERSIST_ROOT/recover_renderer.sh` 拉起，默认 `127.0.0.1:41451` |
| ckpt | `experiments/aerial/rl/artifacts/` 下 WM / actor / depth / tau（与 Phase-2 mainline 一致） |

---

## 1. 一键演示

```bash
ssh <125-host>
cd ~/aerial-wam-v2
chmod +x experiments/aerial/scripts/run_sim_vgoal_demo_short.sh

export TARGET_CLASS=car
export VGOAL_REPO=~/Projects/aerial-vgoal-wam

# 1) 无 AirSim 栈加载 smoke
./experiments/aerial/scripts/run_sim_vgoal_demo_short.sh mock

# 2) AirSim 起来后 — GT 最近物体 smoke（可选）
./experiments/aerial/scripts/run_sim_vgoal_demo_short.sh gt

# 3) 完整 YOLO 闭环
./experiments/aerial/scripts/run_sim_vgoal_demo_short.sh eval
```

路线：`experiments/aerial/annotations/sim_vgoal_demo_short_route.json`（`long_route_01` 前 ~40 m，朝城区车辆方向）。

---

## 2. 手动等价命令

```bash
source experiments/aerial/scripts/env_4090.sh

python -m experiments.aerial.scripts.wam_vgoal_eval \
  --annotation experiments/aerial/annotations/sim_vgoal_demo_short_route.json \
  --routes 0 --episodes 1 \
  --detector yolo --target-class car \
  --vgoal-repo ~/Projects/aerial-vgoal-wam \
  --max-steps 400 --cruise-speed 6 --success-dist 4 \
  --search-det-steer --reject-far-lock-m 40 \
  --spawn-yaw-acquire-steps 40 --spawn-yaw-acquire-deg 90 \
  --planner \
  --traj-out artifacts/sim_vgoal_demo_short/traj \
  --perception-log artifacts/sim_vgoal_demo_short/perception \
  --out artifacts/sim_vgoal_demo_short/result.json \
  --device cuda
```

**成功判据**（eval 默认）：视觉锁定后，到目标世界坐标 **≤ 4 m** 或 tracker `ARRIVED`；`arrived_vision=true` 表示全程无 GPS fallback。

---

## 3. 输出与验收

| 路径 | 内容 |
|------|------|
| `artifacts/sim_vgoal_demo_short/result.json` | 汇总：`arrived`, `arrived_vision`, `goal_from`, `min_d_vision` |
| `artifacts/sim_vgoal_demo_short/traj/` | 轨迹 JSON |
| `artifacts/sim_vgoal_demo_short/perception/` | 逐步感知 JSONL |

期望（YOLO eval）：

- `arrived: true`
- `goal_from: "vision"`（非 `search_only` / mixed）
- `det_recall_early` > 0（前 50 步内至少一次检测命中）

若长时间 `search_only`：加大 `--spawn-yaw-acquire-steps`，或先用 `gt` 阶段确认 spawn 朝向有场景物体。

---

## 4. 与真机演示的关系

| | AirSim（本文） | Orin（[`RUNBOOK_orin_vgoal_demo.md`](RUNBOOK_orin_vgoal_demo.md)） |
|--|----------------|---------------------------------------------------------------------|
| 入口 | `wam_vgoal_eval` | `wam_vgoal_deploy` |
| 感知 | AirSim 渲染 + YOLO | USB 相机 + YOLO |
| 控制 | 仿真速度指令 | MAVLink GUIDED |
| 路线 | 短 annotation JSON | 无 GPS 航点，`--demo-short` |

仿真验证通过后再上 Orin 实飞。

---

## 5. 故障排查

| 现象 | 处理 |
|------|------|
| `Connection refused :41451` | 在 125 上跑 `recover_renderer.sh`，确认 `AIRSIM_HOST=127.0.0.1` |
| `--vgoal-repo not found` | clone `aerial-vgoal-wam` 到 `~/Projects/aerial-vgoal-wam` |
| ckpt missing | 从 Mac/artifact 同步 `wm_step_3500.pt` 等 |
| YOLO 无检测 | 降 `--yolo-conf 0.15`；或 `TARGET_CLASS=truck`；或先 `gt` smoke |
| 远距误锁 | 保持 `--reject-far-lock-m 40` |
