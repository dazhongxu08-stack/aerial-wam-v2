# 破局方案 · Soft→Hard 无教师主线 + 教师天花板对照 · 2026-09-25

**四机：** `125`（4090+AirSim）· `84`（AirSim）· `14`（H100 via 125）· `11`（H100 via 125）  
**合同不变：** routes `0,1,3,4` · toward_g · **主张「学会」必须报关罩** · pass = `arrived≥1` ∧ `P0_stick≤1`（R0+R1）· INIT = `022358_best_sr2`  
**不做：** 再叠 curiosity / openside / neargoal-only；不开罩刷 SR 当主结论；不停换奖励旋钮。

---

## 0′. 今晚优先 · Phase-0 几何证伪 / 修复（阻塞 Soft→Hard）

**主航道合同：全程关罩。** 开罩不进主结论。

**2026-09-25 晚：** C1–C4 已停 → 几何证伪 `20260925_geom` = `FAIL_HARD_NEAR_LATENT` →  
改跑 **geomfix**（代价回流 + 近场 WM），四机不停，直到 `READY_SOFT`。

| 臂 | 机 | 任务 | 过关 |
|----|----|------|------|
| **R1** | 125 | `frames_multi2` pack（d_near=5 保 left_near）→ 训代价头 → **gate `--require-left-near`** → cone rank；AirSim top-up | gate 不过门不 skip；cone PASS |
| **R2** | 84 | 并行 GT left_near 采集（多轮）→ 同步 125 | 持续产出 frames |
| **R3** | 14 | 近场 WM FT（depth-aux+hinge）+ B′-1 | three_zone `forward_min` R²≥0.3 |
| **R4** | 11 | 更强 hinge WM + imagine/cone | imagine gap≥0.05；辅测 R1 cone |

```bash
STAMP=20260925_geomfix SSH125=cursor-125-public \
  bash experiments/aerial/scripts/deploy_geomfix_quad.sh
```

出门：`experiments/aerial/rl/artifacts/geomfix_${STAMP}/PHASE0_VERDICT.md`  
**仅当 `READY_SOFT.txt` 出现** → 才允许 §3 Soft→Hard（仍 `--no-shield`）。

---

## 0. 困境一句话

| 已成立 | 仍卡 |
|--------|------|
| 障碍代价头过门（三组排序，含左近前空） | 关罩 R0/R1 **贴墙**，中程出局 |
| `best_sr2` 罩关中位 **2/4**（R3/R4） | R1 进度 ~28–36%，远低于 V5 的 58% |
| V5（face/peel+开罩）天花板 **3/4** | +160 续训 **塌回 0/4**（过训） |
| B 轮 curiosity/floor/openside/neargoal **全 0/4 证伪** | 再加 shaping 项不会破局 |

**根因判断（要证伪，不要再猜）：**  
关罩硬终止 → 正例太稀 → 策略学不会「侧向离开墙」；V5 的 peel 是手写教师；当前 C 轮把 OL-BC / escape / antistick / terminal 四刀并行，**没有 Soft→Hard 课表，也没有把代价头回流到新状态**。公开资料里 MasterRacing / ProbColl / SIGN 指出的破法是：**先软碰撞攒正例 → 再硬碰撞收紧；代价跟动作走；训不挂盾、验分开关罩。**

---

## 1. 本方案主张（两条线，禁止混报）

| 线 | 主张 | 过关 |
|----|------|------|
| **L-主 · 无教师** | Soft→Hard + 冻结代价头 + early-stop，关罩 SR 中位 ≥2/4 且 R1 prog 中位 ≥50%，P0_stick≤1 | 可对外说「模型学会绕」 |
| **L-天花板 · 弱教师** | OL face/peel 蒸馏（或 V5 栈）开罩 SR≥3/4 | **只作上界**，不得写成无教师成功 |

今早 C1–C4 若还在跑：**等本轮 gate 出数字后立刻停链**（`chain_c1234_to_ab` 不要自动进 A/B 奖励矩阵）。用下面 Phase-1 数字决定砍哪条臂。

---

## 2. Phase-1 · 诊断矩阵（4 机并行，~2–4 h，先做）

全部 **eval only**，同一 actor=`022358_best_sr2`，同一 anno=`outdoor_complex_focus134_inland`，`max_steps=600`，**n=3 seed**。

| 机 | 臂 | Planner | Shield | 手写偏置 | 读什么 |
|----|-----|---------|--------|----------|--------|
| **125** | D0 | closed_loop H=15 | **OFF** | 关 | 关罩基线（应 ≈2/4） |
| **84** | D1 | closed_loop H=15 | **ON** | 关 | 罩是否只抬 R3/R4 |
| **14** | D2 | **open_loop** H=15 | **OFF** | **face/peel 开** | peel 无罩是否降贴墙 |
| **11** | D3 | open_loop H=5 | **ON** | face/peel（V5 全栈） | 天花板 3/4 是否仍在 |

**决策树（Phase-1 出门就执行，禁止拖延）：**

```
D2 关罩 severe/P0_stick 相对 D0 明显下降？
  YES → 走 §3.A「蒸馏 peel」（L-天花板）+ §3.B Soft→Hard 并行
  NO  → 砍蒸馏主投入，全力 §3.B Soft→Hard（L-主）

D0 复现不了 ≥2/4？
  → 先修 eval/ckpt 路径，禁止开训

D3 < 3/4？
  → V5 栈漂移，先复现天花板再谈突破
```

脚本出口（写进 `artifacts/breakout_phase1_${STAMP}/SUMMARY.md`）：每臂 SR / P0_stick / R0·R1 d_final / severe。

---

## 3. Phase-2 · 训练（按决策树只开需要的臂）

### 3.A · L-天花板（弱教师）— 建议机：**125**

- INIT=`best_sr2`；`--planner-rollout open_loop`；`--enable-bc` 蒸馏 peel/escape 候选（可复用今晚 C1，但 **iters≤80**，early-stop 看关罩 gate）
- **每 20 iter** 跑一次关罩 gate n=1；**连续两次关罩 SR=0 或 R1 变差 → 停**
- 验收：关罩 n≥3 与开罩 n≥3 **分表报告**；开罩≥3/4 只标「天花板」

### 3.B · L-主 Soft→Hard（无教师）— 建议机：**84 + 14**

抄 MasterRacing，接到现有 `train_v4_ac`：

| 阶段 | 机 | 设定 | 目的 |
|------|-----|------|------|
| **Soft** | **84** | 关罩；碰撞 **不立刻 terminate**（或 terminate 但 `collided` 罚轻 + 允许穿透 N 步）；障碍代价头 **冻结**；`--no-planner` 采集；想象 AC 开 | 攒「侧向离开墙」正例 |
| **Hard** | **14** | Soft 最好 ckpt → 硬碰撞终止 + 满额 `collided` 罚；仍关罩；仍冻代价头 | 收紧成可部署策略 |
| **验收** | **125** | Soft/Hard 各 ckpt：关罩 closed_loop n=3；辅开罩 | 主结论只看关罩 |

**硬约束：**

1. Soft 阶段 **禁止** 把罩写进奖励（SIGN/NavRL 合同）。  
2. Soft 未出现「R0 或 R1 单局 progress≥0.5 且未贴死」≥ K 条（建议 K=8）→ **不准进 Hard**。  
3. Hard 每 20 iter early-stop；复制 +160 塌缩模式（熵↓ + R3/R4 丢失）→ **回滚 Soft 最好**，禁止盲续。  
4. 语料仍 inland/interior；禁止开阔水岸到达混入。

### 3.C · 代价头回流（ProbColl/ORACLE）— 建议机：**11**

- 用 Soft/Hard 新轨迹 + GT 深度，重标 `(feature, action)→cost`（含侧向楔）  
- **先跑三组过门**（前空 / 前近 / 左近前空）；不过门 **禁止** 换策略起点  
- 过门后再开一轮短 Hard（仍在 14 或回 84）

---

## 4. 四机时间表（墙钟）

```
T0–T4h   Phase-1 诊断（四机并行 eval）
T4h      写 SUMMARY + 决策树结果；kill 无效 C/A/B 链
T4–T16h  Soft@84 ‖ 天花板蒸馏@125 ‖ 代价回流数据准备@11
T16h     Soft 正例门闩检查 → 过则 Hard@14，不过则加长 Soft / 降碰撞罚再一轮
T16–T28h Hard@14 + 125 间歇关罩 gate
T28h     主结论表：关罩 n=3 Soft/Hard vs D0 vs V5；另表开罩
```

`84` 若 Cloudflare 登不上：把 Soft 挪到 **125**，天花板蒸馏挪到 **14**（改 yaml 远程 AirSim 或本地），**不要空转等 84**。

---

## 5. 杀局 / 成功标准

| 结果 | 条件 | 下一步 |
|------|------|--------|
| **主线胜利** | Hard 关罩中位 SR≥2/4 ∧ R1 prog≥50% ∧ P0_stick≤1（n≥3） | 冻 ckpt；录 R0/R1 双视角；进 W2 Demo 栈 |
| **部分胜利** | Soft 有正例但 Hard 关罩仍 <2/4 | 代价回流后再 Hard；或接受「关罩 2/4 + 开罩 3/4」并诚实写 shield 贡献 |
| **天花板-only** | 仅 OL-BC/V5 开罩 ≥3/4，关罩无提升 | **停止宣称无教师学会**；产品走 V5，研究回头修代价/Soft |
| **失败** | Soft 48h 仍无 R0/R1 正例 | 停训；改诊断：spawn/深度标签/动作缩放，而不是再加 reward 项 |

---

## 6. 与今晚已跑作业的关系

- **C1–C4 + `chain_c1234_to_ab`：2026-09-25 晚已停**，改跑 §0′ 几何证伪（`deploy_geom_verify_quad.sh`）。中途 ckpt 仅作附录，不进 Soft INIT。  
- **A/B PathExpert 矩阵**：本破局周 **冻结**（与「无教师主线」冲突；需要时只做附录对照）。  
- Soft→Hard / Phase-1 eval：**等** `geom_verify_*/SUMMARY.md` 为 `PASS_GEOMETRY` 再开。

---

## 7. 开工命令（Phase-1，STAMP 自定）

```bash
# 在 Mac 上（示例）：四机并行 eval — 具体 yaml/脚本以仓库现有
# run_mainchannel_*_noshield / wam_phase2_long_eval 为准，统一 INIT=best_sr2
STAMP=20260926_breakout
# 125 D0 / 84 D1 / 14 D2 / 11 D3  — 由 deploy 脚本一次拉起
# 出门写：artifacts/breakout_phase1_${STAMP}/SUMMARY.md
```

Phase-1 脚本若缺：优先复用 `wam_phase2_long_eval` + focus134 inland + 各机已有 `run_mainchannel_*` 开关（`--no-shield` / planner open_loop / face-peel flags），**不要新开第三条奖励线**。

---

## 8. 一句话

**先用四机 4 小时分清「peel 有没有用、罩有没有用、天花板还在不在」；再用 84/14 做 Soft→Hard 无教师主线，125 做蒸馏上界与验收，11 做代价回流——停止用新奖励项碰运气。**
