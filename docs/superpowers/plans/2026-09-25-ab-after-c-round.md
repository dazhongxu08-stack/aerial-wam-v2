# 过夜接力 · C 轮结束后切 A/B 矩阵 · 2026-09-25

**约定：** stamp `20260925_2100` 的 C1–C4 **跑完 gate 后**自动切本矩阵；不中途杀 C。

## 方法矩阵

| 臂 | 机 | 方法 | 内容 |
|----|----|------|------|
| **A1** | 125 | 自发 | antistick 奖励（press/center），关罩，无折线 |
| **A2** | 84 | 自发对照 | 更强 press/center，其余同 A1 |
| **B1** | 14 | 路径监督 | PathExpert（A*/折线老师）采 interior 成功绕障 → BC → 关罩 gate |
| **B2** | 11 | 路径→去特权 | 同上 BC；**验收只用 toward_g 关罩**（无线） |

合同：routes 0,1,3,4 · INIT floor `022358` · pass = arrived≥1 ∧ P0_stick≤1。  
B 采集遵守 urban-interior：`require interior`、inland 门、`AUTO_REVIEW`、gate 未过继续 append。

## 触发

`chain_c1234_to_ab.sh` 等 `c1234_oa_terminal_${STAMP}_ALL_DONE.txt` → 写 C 摘要 → `deploy_ab1234_spontaneous_path.sh`。
