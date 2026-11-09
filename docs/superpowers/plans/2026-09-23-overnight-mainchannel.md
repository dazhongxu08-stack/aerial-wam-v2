# 过夜主航道自修 · 2026-09-23

**立场：** 目标行为 = **朝终点走主空域/街谷自由通道，遇障绕开，不贴墙**。  
「贴墙」是要消灭的失败，不是策略。「主航道」= 这条行为线，不是 GT 折线产品路径（折线仅作诊断）。

---

## 已证伪（今晚之前）

Anti-stick 矩阵 `20260922_232605`：

| 假设 | 结果 |
|------|------|
| H1：CL actor 上开回 open_loop peel → R0/R1 severe↓ | **否** — 125/84 均为 P0=2/2 |
| H2：仅罩能消贴墙 | **否** — open+罩仍 2/2；closed+罩仅 R0 无 severe 标签，仍停 ~29m |
| 封存 V5 可直接当 inland 天花板 | **弱** — inland 上 V5 也 P0=2/2、SR 2/4 |

结论：不是「评测忘开 peel」；CL 权重/栈不能执行绕障。**下一步不是加 H / 再灌 climb。**

---

## 今晚自动修（并行）

### Phase A — 主航道诊断（125，eval only）

同 actor 对照：`toward_g`（已有）vs **`polyline` + rolling_global**（特权街谷通道）。

| 臂 | Actor | Subgoal | 目的 |
|----|-------|---------|------|
| POLY-V5 | hbclear | polyline+RG | 街谷是否可飞通 |
| POLY-CL | CL 172751 best | polyline+RG | CL 在有胡萝卜时是否仍贴墙 |

**读数：** R0/R1 severe + `d_final`。  
- Poly 好、toward_g 差 → 意图/carrot 问题，进场可上走廊混合。  
- Poly 也差 → 执行/感知问题，必须蒸馏/重训。

### Phase B — open_loop teacher + BC 蒸馏（84 先，125 诊断完接上）

相对已失败的 **closed_loop** CL-FT：训练时 **`--planner-rollout open_loop`** + `--enable-bc`，热启 **hbclear**（不是污染 CL）。

| 机 | 内容 |
|----|------|
| **84** | `ol_bc_ft` 过夜（iters≈40）→ 完后 P0 eval（open±罩各 1 seed） |
| **125** | Phase A 完后同款 `ol_bc_ft` |

**门闩（与 anti-stick 同）：** R0+R1 `severe` 相对 closed CL 基线下降；反指标 near-climb% 飙高不算赢。

### Phase C — 按结果分支（agent 循环决策）

1. **ol_bc 过 P0** → 补 n=3；可选 rem&lt;25 走廊混合（P1）。  
2. **ol_bc 不过、Poly 过** → 实现 mid-course / rem 走廊 subgoal 混合，短 eval。  
3. **两边都不过** → 停训，拉 R0/R1 双视角 + 轨迹取证，改奖励/候选（不盲加 H）。

---



### Phase C′ — R1 开侧偏置（过夜取证后）

R1：`offer` 开但 `d_right→0`、`d_left` 开阔；`wm_bare` 去掉了有用的 `escape_score_w`。
下一刀：**`planner-mock wm_escape`**（WM + 仅开侧 escape/stuck，无 face/strafe/force-peel）。
门闩同 P0；若过 → n=3；若不过 → clearance-centering 奖励 / expert，仍不加重 force-peel。

## Agent 不停

本地 loop ~20min：读两机日志 → 写 `artifacts/overnight_mainchannel_STATUS.md` → 推进下一阶段（启 FT / 启 eval / 取证）→ 不空等。

---

## 一句话

**Anti-stick 证明「开关 peel」不够；今晚 125 诊主航道（poly），84/125 用 open_loop+BC 从 hbclear 重蒸馏绕障，按门闩自动分支。**
