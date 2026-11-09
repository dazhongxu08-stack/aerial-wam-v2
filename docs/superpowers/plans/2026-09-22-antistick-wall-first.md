# 方案 · 贴墙优先（Anti-stick）· 2026-09-22

**立场：** 贴墙 / mid-course `severe` 是 P0。近目标合拢、climb%、25m 候选、加长 H 都排在贴墙门闩之后。允许 climb，但不当主解（V5 证明平飞 + face/peel + 罩即可）。

**证据摘要：**
- 125/84 视频：R0/R1 机头贴立面、街谷夹死；多数局进不了进场。
- A/B/C（25m 强制候选）全 0/4，且 B 更差 → 不是「接近段缺候选」。
- 封存 V5：`open_loop` face/peel + shield，SR 3/4，平飞。

---

## 目标与门闩

| 优先 | 指标 | 通过条件（建议） |
|------|------|------------------|
| **P0** | R0+R1 的 `severe_collision` | 对照相对当前 CL closed+noshield：**severe 明显下降**（理想：两路由合计 severe ≤ 1/2，且 FPV/轨迹不再贴墙） |
| P0 辅 | R0/R1 `d_final` / progress | 能走出中程，而不是 &lt;100 步出局 |
| P1（P0 过后再看） | R3/R4 `d_final`、SR | 进场合拢；不把 climb@near 当 KPI |

**反指标：** 成功或接近成功时 near-climb% 系统性飙高 → 记为「过度依赖 climb」，不记作胜利。

---

## 假设（要证伪）

1. **H1：** 同一 CL actor 上恢复 **`open_loop` face/peel**（手写偏置开）→ R0/R1 贴墙/severe 下降（罩关也该看到能力差）。
2. **H2：** 仅 **`closed_loop` + shield** 不够消灭贴墙（84 shield 三 seed 仍 R0/1/3 severe）→ 罩是网，不是中程策略。
3. **H3：** **V5 全栈**（hbclear + open_loop H=5 + shield）仍是天花板；CL 权重若在 open_loop 下仍贴墙 → 要重训/蒸馏 peel，而不是加 H。

---

## 实验矩阵（同 routes 0,1,3,4 · toward_g · max_steps 600）

| ID | Actor | Planner | Shield | 机器 | 目的 |
|----|-------|---------|--------|------|------|
| **V5** | hbclear latest | open_loop H=5 | ON | **125** | 天花板复现 |
| **CL-A** | 125 CL BEST `172751` | closed_loop H=15 | OFF | **125** | 贴墙基线（已知差，可 1 seed） |
| **CL-B** | 125 CL BEST | **open_loop** H=15 | OFF | **125** | H1：只开 peel，无罩 |
| **CL-C** | 125 CL BEST | **open_loop** H=15 | ON | **125** | peel + 罩 |
| **84-A** | 84 CL BEST `175214` | closed_loop H=15 | OFF | **84** | 84 贴墙基线（可跳过若与已有 noshield 数字一致） |
| **84-B** | 84 CL BEST | **open_loop** H=15 | OFF | **84** | H1 on 84 |
| **84-C** | 84 CL BEST | **open_loop** H=15 | ON | **84** | peel + 罩 |
| **84-D** | 84 CL BEST | closed_loop H=15 | ON | **84** | H2：仅罩 |

每臂默认 **1 seed** 先看效应；若 CL-B/84-B 出现 severe 下降，再补 n=3。

**不做（本轮）：** 加长 H、再灌 25m climb 强制、新开一轮长 CL-FT、以 oc↑/climb% 选 BEST。

---

## 机器分工

```
125 (AirSim + hbclear/V5 ckpt)
  └─ 串行：V5 → CL-B → CL-C   （CL-A 可用已有 cl_noshield_b0 数字，不必重跑）
  └─ 每臂结束：抽 R0/R1 双视角（skip-airsim 即可）看是否仍贴墙

84 (AirSim + CL 175214)
  └─ 串行：84-B → 84-C → 84-D   （84-A 用已有 noshield b0）
  └─ 与 125 并行，缩短墙钟时间
```

两机都只跑 **eval**，不抢训练；84 若仍有旧 resume 矩阵，先停掉再开本矩阵。

---

## 读数模板（每臂）

```
tag | SR | severe_R0 | severe_R1 | d_f_R0 | d_f_R1 | notes_fpv_or_traj
```

判定：
- **CL-B/84-B 相对 closed+noshield：** R0/R1 severe↓ → H1 成立 → 下一阶段把 face/peel 行为蒸馏进 actor（或训练开 open_loop teacher），而不是加 climb。
- **仅 84-D 好、B 不好：** 依赖罩，能力仍未过墙 → 部署可开罩，但宣称「学会绕障」不成立。
- **V5 好、CL open 仍贴墙：** actor/WM 栈与 V5 不兼容 peel → 从 hbclear 热启或 BC peel，而不是加 H。

---

## P0 通过后的 P1（本稿只立项，不本轮跑）

1. 近目标：`rem&lt;25` 切走廊/polyline subgoal（主航道进场）。
2. 训练：online BC / 奖励对准「不贴墙 + 到达」，climb 仅作 jam-band 合法自由度。
3. 对照：open_loop teacher vs closed_loop student 的 severe 差。

---

## 脚本与产物

- 启动：`experiments/aerial/scripts/run_antistick_wall_first.sh`（`HOST=125|84`）。
- 产物：`artifacts/antistick_<stamp>_<host>/` 下各臂 `eval_all.json` + `traj/` + 可选 `videos_skipas/`。
- 日志：`artifacts/logs/antistick_<stamp>_<host>.log`。

---

## 一句话

**先证明「开回 face/peel（±罩）能否去掉贴墙」；125 跑 V5 天花板 + CL open；84 并行跑 CL open/shield 分解。贴墙门闩不过，不谈近目标和 climb。**
