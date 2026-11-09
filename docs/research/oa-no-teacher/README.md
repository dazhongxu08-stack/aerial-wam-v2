# 无教师主动绕障 — 公开资料下载与逐篇分析

**日期**: 2026-09-25  
**对照目标**: Aerial WAM v2 — 视觉世界模型 + 想象规划，**主动学会绕障**；动作条件障碍代价；**关罩训 / 开罩验**；城市硬路线；**无 PathExpert / BC 教师**。

## 目录

| 路径 | 内容 |
|------|------|
| `pdfs/` | 已下载 PDF（16 篇） |
| `txt/` | 抽取文本（供检索） |
| `notes/batch-A.md` | 自监督 / 碰撞预测族 |
| `notes/batch-B.md` | 纯 RL 无人机绕障族 |
| `notes/batch-C.md` | MonoMPC / Dreamer 导航 / GRaD-Nav |
| `notes/npe-drl.md` | NPE-DRL（IEEE，无 arXiv；按仓库 README） |

误下已删：`npe_drl.pdf` 曾错成轮腿摩擦论文。

---

## 总排序（对我们最有用）

| 优先级 | 工作 | 无教师 | 最该抄什么 |
|--------|------|--------|------------|
| **P0** | ProbColl / ORACLE / BADGR | 是 | `(观测, 动作序列)→碰撞/代价`；训时随机/关安全采数据，部署用代价过滤 |
| **P0** | MonoMPC | 半（LiDAR 特权标） | clearance **分布** + 风险校准；噪声深度当条件不当几何 |
| **P0** | SIGN / NavRL | 是（策略） | **训练不挂盾、部署挂盾**；动作条件 `Qc` / VO |
| **P1** | MasterRacing / DiffSim-Avoidance | 是 | Soft→Hard 碰撞课表；无专家梯度训 |
| **P1** | TurtleBot-DreamerV3 | 是 | WM+imagination 无教师导航先例（但 LIDAR） |
| **P1** | Future Depth | 是 | 用未来深度当稠密价值，少撞也能学 |
| **P2** | MAVRL / AvoidBench | 是 | 记忆 latent、变速度、评测套件 |
| **P2** | Collision Replay | 是 | 撞后回放作标签增广 |
| **P3** | GRaD-Nav / Swarm E2E / Self-Learning CAS | 部分/是 | 旁证；栈差较大 |
| **对照** | NPE-DRL | **否**（APF 非专家教师） | 仅作「有弱教师」对照，**非我们主线** |

**公开空白（可写成贡献）**: 几乎没有「**单目视觉 RSSM 想象 + 动作条件障碍代价 + 关罩训开罩验 + 城市楼群**」同时做满的先例。

---

## 建议落地配方（无专家）

1. **障碍头**: ProbColl/ORACLE 式 — `(feature_t, action)` 或短序列 → 碰撞/clearance；标签用仿真 `collided` + 未来 GT min-depth（特权自监督，非专家）。
2. **过门**: 前空直飞最高 / 前近朝墙低于朝空 / 左近前空则往前高于往左（你们已有）。
3. **策略**: 关罩纯 RL + Soft→Hard（MasterRacing）；可选 Future Depth 辅助。
4. **部署**: NavRL/SIGN 式开罩；主张「会了」必须报罩关数字。
5. **不做**: PathExpert/BC；NPE 的 APF 教师（除非只做消融对照）。

---

## PDF 清单

| 文件 | 论文 |
|------|------|
| `probcoll.pdf` | Uncertainty-Aware RL for Collision Avoidance |
| `badgr.pdf` | BADGR |
| `collision_replay.pdf` | Collision Replay |
| `future_depth.pdf` | Future Depth as Value Signal |
| `self_learning_cas.pdf` | Self-Learning Collision Avoidance (thesis) |
| `oracle_collision_pred.pdf` | Motion Primitives + Deep Collision Prediction |
| `mavrl.pdf` | MAVRL |
| `navrl.pdf` | NavRL |
| `masterracing.pdf` | MasterRacing (soft→hard) |
| `avoidbench.pdf` | AvoidBench |
| `swarm_e2e_drl.pdf` | Quadrotor Swarm E2E DRL |
| `diff_sim_avoidance.pdf` | Diff. Sim. depth→bodyrate |
| `sign.pdf` | SIGN |
| `monompc.pdf` | MonoMPC |
| `turtlebot_dreamerv3.pdf` | DreamerV3 TurtleBot navigation |
| `grad_nav.pdf` | GRaD-Nav（抽取见 `txt/grad_nav.txt` from HTML） |
