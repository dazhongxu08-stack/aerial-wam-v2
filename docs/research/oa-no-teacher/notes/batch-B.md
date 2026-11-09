# Batch B — 无教师障碍回避相关论文笔记

目标对照（Aerial WAM v2）：视觉世界模型 + imagination planner **从交互中主动学 OA**，无 expert/teacher demo；当前路径是 **action-conditioned obstacle cost on latents**；**训练关罩、部署开罩**；Urban AirSim 硬路由。

卡点默认指：稀疏碰撞信号、贴墙/卡住、关罩训练却指望部署罩兜底、latent imagination 里障碍成本是否可学、urban 几何困难场景。

---

### MAVRL — MAVRL: Learn to Fly in Cluttered Environments with Varying Speed
- **出处**: Yu, De Wagter, de Croon (TU Delft); arXiv:2402.08381, 2024
- **无教师？**: 是（纯 PPO + 自采深度数据训 VAE/LSTM；无 expert demo / teacher distillation）
- **方法摘要**:
  - Depth → VAE latent → LSTM 记忆增强 latent（显式重建 past/current/future depth）→ 拼状态/目标 → PPO 出 body-frame 加速度 + yaw rate；下游 Agilicious MPC
  - 分阶段：先无障碍 PPO 采深度 → 训 VAE → 冻 encoder 训 LSTM → 冻表征再训 PPO（复杂障碍）
  - **变速度**：到达奖励与 TRAV（环境可通行性）反相关 + 超速惩罚，复杂区减速、空旷区加速
  - AvoidBench 仿真；实机 minimal fine-tune
- **传感/动作**: 立体深度（AvoidBench SGM，仿真实深度噪声）；加速度 + yaw rate → MPC body rate/thrust
- **障碍如何进入学习**: 碰撞惩罚 `rcollision` + 边界/到达稀疏项 + progress shaping；障碍几何经 depth→latent 间接进入；无显式 ESDF/occupancy 奖励项
- **与我们卡点的关系**:
  - **记忆 latent** 直接对口「FOV 外挡墙后回看/贴大墙卡住」——大障碍出视野后仍记得
  - 变速度与我们 antistick/关罩贴墙相关：固定高速易撞，无记忆易 stuck
  - 仍是 model-free PPO，**无 world-model imagination**；碰撞仍是稀疏散励
- **可抄实现点**:
  - LSTM 对 past/current/future depth 加权重建（ablation：偏 past+current 优于只预测 future）
  - TRAV 耦合的到达奖励做 safety–agility 平衡
  - 先无障碍采深度再冻表征训策略（表征与策略解耦 timeline）
- **局限/不适配**: 依赖 depth（非纯 RGB monocular 合同）；无开源仓库提及；仿真是森林/室内柱，非长程 urban goal；无 shield train/deploy 分离
- **推荐优先级**: 中（记忆 latent + 变速 shaping 可抄；整体栈不是 WM）

开源: 文中未给 repo（依赖 AvoidBench: https://github.com/tudelft/AvoidBench）

---

### NavRL — NavRL: Learning Safe Flight in Dynamic Environments
- **出处**: Xu et al. (CMU); RA-L 2025; arXiv:2409.15634
- **无教师？**: 是（Isaac Sim 并行 PPO；无 imitation / privileged teacher）
- **方法摘要**:
  - 感知：静态 occupancy voxel + 动态 bbox/速度（U-depth + DBSCAN 集成，YOLO 分动静）
  - 状态：goal-frame 相对位姿/速度 + 动态障碍矩阵 + **360° 射线距离图**（非 raw 图）→ CNN 抽特征 → PPO
  - 动作：Beta 分布归一化速度命令（goal frame），可调 `vlim`
  - **训练后部署加 VO safety shield**：不安全速度投影到 VO 外（LP 风格）
  - 障碍密度 curriculum；数千机并行
- **传感/动作**: RGB-D → 结构化障碍表征；输出 3D 速度
- **障碍如何进入学习**: 稠密 shaping——`rss = mean log(ray_dist)`、`rds = mean log(dist_to_dyn)` + 碰撞 reset；部署再用 VO shield 纠动作
- **与我们卡点的关系**:
  - **Train 学策略 / Deploy 加 shield** 与我们「关罩训、开罩部署」同构——文中明确 black-box 需 shield
  - 射线距离奖励 ≈ 显式 clearance/antistick，比稀疏碰撞更可学
  - 状态已是 obstacle-centric，**不是 latent WM imagination**；sim2real 靠表征而非视觉世界模型
- **可抄实现点**:
  - 静态安全奖励用 log 射线距（可迁移到我们 obstacle cost / clearance term）
  - VO shield 作部署兜底（对应我们 ThreeZone，但他们是速度空间投影）
  - Beta 动作分布 + goal-frame 坐标减少全局系依赖
  - 密度 curriculum
- **局限/不适配**: 重度依赖 voxel/检测栈与 depth；动态障碍为主卖点，urban 长程 monocular 不对口；无 imagination planner
- **推荐优先级**: 高（shield 训练–部署分离 + 稠密 clearance 奖励最贴我们卡点）

开源: https://github.com/Zhefan-Xu/NavRL ；视频 https://youtu.be/EbeJW8-YlvI

---

### MasterRacing — Mastering Diverse, Unknown, and Cluttered Tracks for Robust Vision-Based Drone Racing
- **出处**: Yu, Hu, Su, Deng, Zhang, Zou (SJTU); RA-L preprint 2025; arXiv:2512.09571
- **无教师？**: 部分（无 IL/teacher distillation；但 actor 有噪声 gate 命令，critic 用 **特权真值 gate 命令**——asymmetric AC；非纯视觉无特权）
- **方法摘要**:
  - **两阶段碰撞**：Soft（穿障无终止，碰撞点穿透计 mild 罚）→ Hard（刚体碰撞终止 + 重罚）——先学高速竞速再学避障
  - Adaptive noise curriculum 污染 next-gate 命令 + asymmetric actor-critic，逼 actor 从 depth 认 gate
  - Track primitive 生成器（圆/折线/椭圆）+ L2C2 Lipschitz 平滑 + DiffLab (Isaac Lab)
  - Depth CNN + 状态 MLP → CTBR
- **传感/动作**: 深度 96×72 + 噪声相对门位置；CTBR；实机 RealSense + Vicon
- **障碍如何进入学习**: Soft 阶段穿透点计数罚；Hard 阶段碰撞终止/罚；障碍只经 depth + 碰撞几何，无 ESDF 奖励；门/障碍 depth 易混淆靠 curriculum 解
- **与我们卡点的关系**:
  - **Soft→Hard 碰撞** 直接回应「早终止导致学不动避障 / 探索–利用冲突」——我们关罩时也类似「先能飞再罚撞」
  - Soft 不终止长 rollout ≈ 放大正样本覆盖，对贴墙卡死探索有启发
  - 任务是 racing+门，非 urban goal；有特权 critic，严格「无教师」不成立
- **可抄实现点**:
  - Soft-collision（穿透计数、不终止）再 hard 精修
  - 性能驱动的 adaptive noise curriculum（成功加噪、失败降噪）
  - L2C2 动作平滑正则（减抖动）
- **局限/不适配**: 门任务 + 特权 critic；DiffLab 竞速动力学；无 WM；urban 几何「死胡同」未覆盖
- **推荐优先级**: 高（soft/hard 两阶段碰撞对「无教师学 OA」训练动力学极有用）

开源: https://github.com/MasterDroneRacing/MasterRacing ；DiffLab 同仓库生态

---

### AvoidBench — AvoidBench: A high-fidelity vision-based obstacle avoidance benchmarking suite for multi-rotors
- **出处**: Yu, de Croon, De Wagter (TU Delft); arXiv:2301.07430, 2023
- **无教师？**: N/A（基准/评测套件，不是学 OA 算法）
- **方法摘要**:
  - Flightmare + Unity 照片级场景 + RotorS 动力学；Python/C++ 插件接口
  - 森林/室内仓；Poisson Disc 控障碍间距；可改纹理不透明度等
  - **环境指标**：Traversability（TRAV）、Relative Gap Size
  - **性能指标**：SR、Path Optimality、Energy(jerk)、AGV、Mission Progress、Processing Time 等
  - 对比 Ego-planner / MBPlanner / Agile-autonomy，并做实机趋势验证
- **传感/动作**: 面向视觉 OA 算法通用接口（深度/图像由场景提供）
- **障碍如何进入学习**: 不进入学习；障碍进入**评测复杂度定义**（TRAV、gap）
- **与我们卡点的关系**:
  - TRAV/gap 可作 hard-route 分层与论文表格口径，补 AirSim urban 难度轴
  - MAVRL 直接建在 AvoidBench 上——批内可串联
  - 不解决无教师学习本身
- **可抄实现点**:
  - TRAV + Relative Gap 定义难度档
  - Mission Progress / Path Optimality 与我们 Prog/SPL 对齐写法
  - 程序化障碍场 + 多 trial seed
- **局限/不适配**: 非学习方法；场景非长程 urban interior；不涉及 WM/shield
- **推荐优先级**: 低（评测工具；写论文/难度分层时中用，对学 OA 低）

开源: https://github.com/tudelft/AvoidBench

---

### Swarm-E2E-DRL — Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning
- **出处**: Huang, Yang, Krupani, Şenbaşlar, Batra, Sukhatme (USC); arXiv:2309.13285v2, 2024
- **无教师？**: 是（end-to-end DRL → 转子推力；无 IL；可与经典规划器对比）
- **方法摘要**:
  - 去中心化策略：自状态+邻居+**SDF 风格障碍观测**（3×3 最近距网格，排列不变）→ MLP + multi-head attention → 四转子推力
  - **碰撞片段 replay**：撞前 1.5s 状态入 buffer，按 `αr` 重放，超过重放次数剔除「当前学不动」的难例
  - 奖励：距目标 + 机–障/机–机碰撞指示（每撞只罚一次）+ 近邻接近罚 + 控制/姿态正则
  - Sample Factory 异步 IPPO；Crazyflie 零样本；sim 到 32 机 / 80% 障碍密度
- **传感/动作**: 状态级 SDF 障碍观测（非 raw vision）；动作 = 归一化推力
- **障碍如何进入学习**: 观测侧 SDF 距离网格；奖励侧 `αocol` 碰撞指示 + replay 放大稀疏撞点
- **与我们卡点的关系**:
  - **Collision replay / clipped episode** 直接打「稀疏碰撞难学」——比 PLR 更专盯避障
  - SDF 观测量少、排列不变，像轻量 obstacle cost 输入；但**不是视觉 WM**
  - 群体 + 室内柱网格，非 monocular urban；需完整状态
- **可抄实现点**:
  - 撞前窗口 replay + 可学性阈值踢出 buffer
  - 碰撞指示「每事件罚一次」减长时接触噪声
  - Attention 融合邻居 vs 障碍优先级
- **局限/不适配**: 无相机深度/RGB；推力端到端与我们速度/想象栈不同；swarm 开销非目标
- **推荐优先级**: 中（replay 机制高价值；传感形态不适配）

开源: 网站 https://sites.google.com/view/obst-avoid-swarm-rl （文中未给 GitHub）

---

### DiffSim-Avoidance — Simple but Stable, Fast and Safe: Achieve End-to-end Control by High-Fidelity Differentiable Simulation
- **出处**: Li, Wang, Huang, Sun, Wu, Yan, Zou, Yu (SJTU 等); arXiv:2604.10548, 2026
- **无教师？**: 是（明确 **without expert guidance / no privileged / no curriculum / no IL**；BPTT via differentiable sim）
- **方法摘要**:
  - Depth（逆深度 + max-pool 到 12×16）+ 航向系速度/角速度/姿态 → **直接 CTBR**；无 map / LSTM / backbone / 外环控制器
  - 高保真可微动力学 + 有限 horizon **BPTT**；解析碰撞奖励梯度回传到图像特征
  - 奖励：速度跟踪 + perception-aware（速度对齐机头）+ 稳定(jerk/ω) + **近障罚** `−clip(drisk−dcol−r)²·detach(v_appr)` + 强碰撞罚
  - **ESDF 梯度重塑**：改最近点几何，把「对撞刹车」偏置改成「侧向绕飞」，缓解大墙/高速 death zone
  - 零样本室外/密林至 ~7.5 m/s；自承迷宫规划弱，适合作前端安全层
- **传感/动作**: 深度图；CTBR 直出飞控
- **障碍如何进入学习**: 可微仿真里最近障碍距的**解析奖励**经 BPTT 监督整条轨迹（几何模态监督视觉策略）；非标签 demo
- **与我们卡点的关系**:
  - **无教师 + 可微碰撞成本** 与我们 latent obstacle cost 哲学最近：障碍成本是解析的，策略端学习
  - 梯度重塑对「贴大墙只刹不绕 / antistick」高度相关
  - 自身承认缺 planning——正好对应我们 imagination planner 的定位（他们当安全控制器）
  - 训练是一阶 BPTT 非 Dreamer；深度输入 vs 我们 RGB 合同有差距
- **可抄实现点**:
  - 近障罚 × 接近速度（detach 速度防乱梯度）
  - ESDF/最近点 **shift** 把梯度从制动改绕行
  - 极简 depth 池化 + perception-aware 对齐
  - 「OA 策略作底层 + 外部分规划」的分工声明（与我们 WM 规划+盾一致）
- **局限/不适配**: 需可微仿真与真机参数辨识；非 RSSM imagination；迷宫/复杂拓扑弱；深度非 RGB
- **推荐优先级**: 高（无教师解析碰撞成本 + 大墙梯度重塑；开源可跟）

开源: https://github.com/Fanxing-LI/avoidance

---

### SIGN — SIGN: Safety-Aware Image-Goal Navigation for Autonomous Drones via Reinforcement Learning
- **出处**: Yan, Huang, He, Guo, Zhao (NUS); arXiv:2508.12394v2, 2025
- **无教师？**: 部分（ImageNav RL 无 expert；**OA 主要靠部署期 depth collision predictor + 动作校正**，训练刻意不把硬安全塞进政策以免探索崩——与我们关罩训同构）
- **方法摘要**:
  - Habitat ImageNav：RGB obs + RGB goal → ResNet/GRU → 连续线速度/角速度；辅任务（图像扰动、未来转移/奖励预测）提样本效率
  - **Safety module**：depth→min-pool 向量 + 动作 → `Qc` 碰撞概率（hard+soft label 混合）；危险时对动作梯度投影迭代修正至 `Qc < dc`
  - 明确：训练允许撞以保探索；安全过滤器部署期启用
  - 无全局地图/外定位；VIO+PID 速度环实机
- **传感/动作**: RGB（导航）+ Depth（安全模块）；连续 `v_lin, v_ang`
- **障碍如何进入学习**: 导航奖励跟 ZER（接近目标图）；碰撞不主导训练；障碍经 **action-conditioned Qc(sd, a)** 在部署纠偏——形态上最像我们「action-conditioned obstacle cost + 关罩训/开罩部署」
- **与我们卡点的关系**:
  - **解耦：导航 RL vs safety filter** 几乎是我们盾策略的论文版论证
  - Soft collision label（距阈值 sigmoid）可改进 obstacle cost 监督平滑度
  - 任务是室内 ImageNav（Gibson/MP3D），非 urban AirSim 长程；无 WM
- **可抄实现点**:
  - `Qc(s_depth, a)` + 梯度投影动作校正（对照我们 shield 干预）
  - hard/soft label 1:1 训碰撞头
  - 「训练不塞硬安全、部署过滤」的实验叙事
- **局限/不适配**: Habitat 低速；安全模块用 depth 真值；非 latent imagination；Image-goal ≠ 坐标 goal
- **推荐优先级**: 高（train/deploy 安全解耦 + action-conditioned 碰撞头，最贴当前工程叙事）

开源: https://github.com/Zichen-Yan/SIGN

---

## 本批结论

按「无教师主动学 OA + 关罩训/开罩部署 + latent/想象成本」对我们的可用度排序：

| 排名 | 短名 | 优先级 | 一句话用途 |
|------|------|--------|------------|
| 1 | **SIGN** | 高 | 训练不管死安全、部署用 `Q(s,a)` 纠动作——直接支撑我们 shield OFF/ON 叙事与 action-conditioned cost |
| 2 | **NavRL** | 高 | 稠密 clearance 奖励 + 部署 VO shield；开源完整可对照实现 |
| 3 | **DiffSim-Avoidance** | 高 | 无教师解析碰撞成本 + 大墙梯度重塑（刹→绕）；可补我们 obstacle cost / antistick |
| 4 | **MasterRacing** | 高 | Soft→Hard 两阶段碰撞，专治早终止学不动 OA |
| 5 | **Swarm-E2E-DRL** | 中 | 碰撞片段 replay 放大稀疏撞信号；传感形态需改写到视觉/latent |
| 6 | **MAVRL** | 中 | 记忆 latent + 变速 TRAV shaping，治大墙出视野卡住 |
| 7 | **AvoidBench** | 低 | 评测/难度轴（TRAV、gap），不教怎么学 |

**本批最该立刻抄进实验的三件事**：
1. **SIGN/NavRL 式解耦**：关罩时用稠密/可微障碍成本学行为，开罩只做审计投影（已有方向，补齐论文对照）。
2. **MasterRacing Soft→Hard** 或 **Swarm collision replay**：解决稀疏碰撞与早终止。
3. **DiffSim 梯度重塑 / NavRL log-clearance**：把「贴墙只压油门」改成「侧向净空」——对齐 antistick 奖励设计。

**对本批「世界模型 + imagination」缺口**：七篇几乎都是 model-free / 可微仿真直出控制，**没有一篇把障碍成本放进 latent imagination rollout**。我们的差异点应继续写：在 RSSM latent 上做 action-conditioned obstacle cost，并用 imagination 选动作，而非仅部署过滤器。
