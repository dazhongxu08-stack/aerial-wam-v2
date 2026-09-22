# Orin 短路径视觉导航演示 Runbook

**目标**：无人机识别场景中指定物体（如车、人），自主规划短路径并导航接近该物体。

**栈**：Phase-2 vgoal（YOLO → TargetTracker → π + WM）+ `wam_vgoal_deploy` + Pixhawk GUIDED。

**硬件**：Orin + Pixhawk 6C + USB 相机 + H12（ch7 Orin 切换，ch8/ch9 录制）。

---

## 0. 演示场景布置

| 项 | 建议 |
|----|------|
| 目标物体 | 选一个 COCO 类，如 `car`、`person`；放在机头前方 **10–25 m**、相机可见 |
| 飞行高度 | 与训练接近，室外 **~5 m**（相对起飞点） |
| 路径长度 | 短演示 **15–25 m** 直线接近即可 |
| 安全 | 开阔场地、无行人；**先卸桨**做 preflight/detect |

---

## 1. 一键演示（Orin 上）

```bash
ssh orin-direct
cd ~/aerial-wam-v2
chmod +x experiments/aerial/scripts/run_orin_vgoal_demo_short.sh

export TARGET_CLASS=car          # 改成你要识别的物体
export VGOAL_REPO=~/Projects/aerial-vgoal-wam

# 阶段 1–3：预检 + 检测 + 加载模型（卸桨）
./experiments/aerial/scripts/run_orin_vgoal_demo_short.sh all

# 阶段 4：实飞（装桨、场地清空后）
./experiments/aerial/scripts/run_orin_vgoal_demo_short.sh fly
```

### 各阶段说明

| 阶段 | 命令 | 作用 |
|------|------|------|
| preflight | `... preflight` | 释放 RC override、STABILIZE；YOLO 单帧探测 |
| detect | `... detect` | 相机连续 40 帧检测，确认目标在视野内 |
| load | `... load` | 加载 WM/π/depth，bench 一次 observe |
| fly | `... fly` | GUIDED 闭环：识别 → 跟踪 → 接近目标 |

---

## 2. 手动命令（等价）

### 2.1 检测台架（不加载 WM）

```bash
python3 -m experiments.aerial.scripts.wam_vgoal_deploy \
  --demo-short --detect-bench \
  --camera 0 --target-class car \
  --vgoal-repo ~/Projects/aerial-vgoal-wam --device cuda
```

### 2.2 短路径闭环实飞

```bash
python3 -m experiments.aerial.scripts.wam_vgoal_deploy \
  --demo-short \
  --mavlink-port /dev/ttyACM0 --camera 0 \
  --vgoal-repo ~/Projects/aerial-vgoal-wam \
  --target-class car \
  --offboard --arm --run \
  --i-know-props-are-on --device cuda
```

`--demo-short` 自动设置：

- `--no-fallback-toward-g`：只跟视觉目标，不用 GPS 航点
- `--success-on-visual`：接近视觉目标 **4 m** 内判成功
- `--search-det-steer`：搜索时朝检测框转向
- `--cruise-speed 4`、`--no-depth-shield`、短路径步数上限 120

### 2.3 监视与录制

```bash
# 无地面站状态
python3 -m experiments.aerial.scripts.orin_fc_monitor

# 录制：默认 ch8 开始 / ch9 停止；或加 --record-auto
```

数据目录：`~/aerial-wam-v2/artifacts/orin_deploy/run_*/`

---

## 3. 演示流水线（每步在做什么）

```
相机帧 → YOLO 检测 target_class
       → 深度反投影 goal_rel
       → TargetTracker（TRACKING / SEARCHING）
       → LatentActorDeployPolicy 输出 body 速度
       → ThreeZone shield（demo 默认关闭）
       → Pixhawk GUIDED 速度指令
```

- **SEARCHING**：慢速前进 + 转向搜索；`--search-det-steer` 会朝检测框转
- **TRACKING**：锁定目标，π 规划接近
- **成功**：视觉目标距离 ≤ `--success-dist`（demo 默认 4 m）

---

## 4. 依赖检查

| 依赖 | 路径 |
|------|------|
| 本仓库 | `~/aerial-wam-v2` |
| vgoal 侧车 | `~/Projects/aerial-vgoal-wam` |
| Python venv | `~/sim_verify/.venv` |
| Checkpoints | `experiments/aerial/rl/artifacts/` 下 WM/π/depth/tau（需已同步到 Orin） |
| YOLO 权重 | `yolov8n.pt`（首次自动下载） |

---

## 5. Mac 远程查看 Orin 相机（MJPEG）

Orin 上跑 MJPEG HTTP 服务，Mac 经 SSH 隧道在浏览器实时看图。

```bash
# Mac 一键：同步脚本 → 启动推流 → 开隧道 → 打开浏览器
chmod +x experiments/aerial/scripts/mac_orin_camera_view.sh
./experiments/aerial/scripts/mac_orin_camera_view.sh

# 同 Wi-Fi 时指定 IP
ORIN_SSH=yao@192.168.1.16 ./experiments/aerial/scripts/mac_orin_camera_view.sh
```

浏览器地址：`http://127.0.0.1:8088/`（快照：`/snapshot.jpg`）

Orin 上手动启动：

```bash
ssh orin-direct   # 或 yao@192.168.55.1 (USB)
~/aerial-wam-v2/experiments/aerial/scripts/run_orin_camera_stream.sh
```

Mac 仅开隧道：`ssh -f -N -L 8088:127.0.0.1:8088 yao@<orin-ip>`

| 变量 | 默认 | 说明 |
|------|------|------|
| `ORIN_SSH` | `yao@192.168.55.1` | 自动 fallback `192.168.1.16` / `10.229.66.164` |
| `CAMERA` | `0` | V4L2 设备（SJCAM USB） |
| `PORT` | `8088` | 仅绑定 Orin `127.0.0.1`，经隧道访问 |

停止：`ssh <orin> 'pkill -f orin_camera_stream'`

---

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| detect-bench 0 hit | 调整目标位置/光照；降低 `--yolo-conf`；换 `TARGET_CLASS` |
| 只搜索不接近 | 目标太远或遮挡；拉近到 20 m 内再飞 |
| 飞到错误方向 | 用了默认 `--fallback-toward-g`；务必 `--demo-short` 或 `--no-fallback-toward-g` |
| 电机不转 | 查电池电压；`orin_reset_h12_control` |
| GUIDED 室内失败 | 必须在室外或有大尺度相对定位；保持 STABILIZE 起飞后再进 GUIDED |

---

## 7. 与 Phase-3 unified 的关系

Phase-3 outdoor **仿真**用几何航点，**不含 YOLO**。实机视觉演示用本文档与 `wam_vgoal_deploy`，不用 `eval_phase3_outdoor_regression_gate.sh`。
