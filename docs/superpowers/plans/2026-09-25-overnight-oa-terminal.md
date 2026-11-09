# 过夜主航道下一刀 · OA + 终端收口 · 2026-09-25

**前提：** stamp `20260925_1007` B 四验证（curiosity / floor→B / openside / neargoal）全 SR 0/4、OA 通道静默 falsify。  
**合同不变：** routes 0,1,3,4 · toward_g · **关罩** gate · pass = `arrived≥1` ∧ `P0_stick≤1`。  
**INIT：** 可复现 floor `022358_best_sr2`（对照 ≥2/4）。

## 四方案（最小差分，一台一刀）

| 臂 | 机 | 方案 | 差分 | 针对 |
|----|----|------|------|------|
| **C1** | 125 | `ol_bc` | open_loop planner teacher + online BC | 把 peel 蒸馏进 actor |
| **C2** | 84 | `wm_escape` | OL + `mock_mode=wm_escape` + BC | 开侧 escape/stuck 教师 |
| **C3** | 14 | `antistick` | press_wall + clearance_center 奖励 | 关罩自学不贴墙 |
| **C4** | 11 | `terminal` | 轻 antistick + 强 success/progress + near_goal 0.5 | 门口收口 + 不丢 OA |

**不做：** 再叠 curiosity / openside / neargoal-only；不开罩刷 SR。

## 门闩

- 主读：`gate_noshield` SR / P0_stick / R0·R1 `d_final`
- 辅：`oa_n_steps` / `offer_escape` 是否从 0 起来（C1/C2 训练期）
- 反指标：near-climb% 系统性飙高不算赢

## 启动

```bash
STAMP=20260925_2100 SSH125=cursor-125-public \
  bash experiments/aerial/scripts/deploy_c1234_oa_terminal.sh
```

通宵看守（125 上）：`watch_c1234_oa_terminal.sh` 死了会重启对应臂。
