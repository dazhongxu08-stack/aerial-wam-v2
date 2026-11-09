# Batch C — 无教师障碍回避相关论文笔记

目标对照（Aerial WAM v2）：视觉世界模型 + imagination planner **从交互中主动学 OA**，无 expert/teacher demo；当前路径是 **action-conditioned obstacle cost on latents**；**训练关罩、部署开罩**；Urban AirSim 硬路由。

卡点默认指：稀疏碰撞信号、贴墙/卡住、关罩训练却指望部署罩兜底、latent imagination 里障碍成本是否可学、urban 几何困难场景。

本批范围：`monompc.txt`、`turtlebot_dreamerv3.txt`、`grad_nav.txt`（及同文 `grad_nav.html`）；`sign.txt` 已在 Batch B，跳过。`npe_drl.txt` 误下，见文末「待替换」。

---

### MonoMPC — MonoMPC: Monocular Vision Based Navigation with Learned Collision Model and Risk-Aware Model Predictive Control
- **出处**: Sharma, Jadhav, Paul, Krishna, Singh (Tartu / IIIT-H / Estonian Aviation Academy); RA-L preprint 2025; arXiv:2508.07387v3
- **无教师？**: **否 / 半监督特权监督**（无 IL/BC 出策略，但碰撞模型用 **LiDAR 真值 clearance** + teleop / 带 GT 碰撞检查的轨迹监督；非纯交互 RL）
- **方法摘要**:
  - 核心主张：DepthAnything 等估计深度**太噪，不能直接做 collision map**；改为把估计点云当作条件，学 **action-conditioned 概率碰撞模型** \(d(u,o)\)：预测给定控制序列上的 **worst-case obstacle clearance** 分布 \((\mu,\sigma)\)
  - 部署：采样式 risk-aware MPC，用 MMD(Maximum Mean Discrepancy) 作 chance-constraint 风险代理，选低风险且到目标的控制序列
  - **Joint / task-aware 训练**：除 Gaussian NLL 外，用 clearance 样本 → 约束违反 → MMD 风险 → MLP+softmax 做 safe/unsafe 二分类 CE；同时学核宽 \(\lambda\)，校准方差，避免过保守/过自信
  - 数据：teleop + 先用 LiDAR GT 跑 Alg.1 采数据训初模，再用学到的模型再采精修；~80k 点云 × 每帧 50 条随机控制 ≈ 6M 训练点
- **传感/动作**: Jackal 地面车；部署 **仅 RGB**（DepthAnything-V2 → 300×2 估计点云）+ 状态 \((x,y,\psi,v,\omega)\)；动作序列 = 线/角速度 horizon \(T{=}5\)s / 50 步；训练用 LiDAR 算 \(d_{gt}\)
- **障碍如何进入学习**: 监督标签 = 轨迹相对 LiDAR 点云的 **min clearance**；障碍经「估计深度条件 + 动作序列」进概率模型，再经 MMD 风险进 MPC——**不是稀疏 reward，也不是 occupancy 直接规划**
- **与我们卡点的关系**:
  - **形态上最贴我们 action-conditioned obstacle cost**：\(o\) + 控制序列 \(u\) → clearance / risk，再进 planner 打分（对照我们 latent cost + imagination）
  - 「噪声深度不当几何、当条件」对 urban monocular / 估计深度合同有说服力
  - **不是无教师 RL**，也 **不是 world-model imagination**：特权 LiDAR 监督 + 已知动力学 MPC；训采仍需安全/不安全轨迹覆盖
  - 无 train-shield OFF / deploy-shield ON 叙事；风险项本身是软安全
- **可抄实现点**:
  - 预测 **clearance 分布** 优于直接二值碰撞概率（监督更易从仿真/测距得到）
  - **下游风险头联合训方差**（NLL + CE），防 obstacle cost 不确定度错位
  - 部署侧采样 MPC / elite set 对控制批次按 risk 再按 goal cost 筛——可对照 imagination 候选打分管线
  - 安全/不安全轨迹联合训 + UR/CR/ECE 校准指标写法
- **局限/不适配**: 地面 2D、室内纸箱 clutter；依赖 foundation depth + PointNet++；训练强依赖 LiDAR 特权；非 aerial / 非 RSSM；无开罩–关罩解耦实验
- **推荐优先级**: **高**（action-conditioned clearance/risk 与我们 obstacle cost 同构；校准联合训可直接抄；需在笔记里标明「非无教师」）

开源: https://github.com/Basant1861/MonoMPC ；网站 https://sites.google.com/view/monompc

---

### TurtleBot-DreamerV3 — World Models for Autonomous Navigation of Terrestrial Robots from LIDAR Observations
- **出处**: Steinmetz, Demo Rosa, Kich, Bottega, Grando, Gamarra (UFSM / Tsukuba / FURG / UTEC); Journal of Intelligent & Fuzzy Systems / arXiv:2512.03429v1, 2025
- **无教师？**: **是**（纯 DreamerV3 交互学习；无 expert demo / IL；无特权 critic 声明）
- **方法摘要**:
  - 在 TurtleBot3 mapless 导航上套 **DreamerV3**：MLP-VAE 把 **全量 360 线 LIDAR** + 相对目标距/角 + 上一步动作压成 latent；RSSM/记忆预测未来 latent、奖励与终止；**imagination rollout** 训 actor-critic
  - 对比 SAC / DDPG / TD3；强调 model-free 直接吃 360 维易崩，latent + 想象更稳
  - 环境 stage 1–6 递增进障碍与尺度；难关用 stage 4→5→6 **预训练迁移**；控制环固定 6.67 Hz 消硬件步频差
  - 结果：全 360 输入时 Dreamer **各环境 100% success**，model-free 普遍 <85%
- **传感/动作**: 仅 LIDAR（10 束 vs 360 束对比）+ 相对目标；连续线/角速度；Gazebo/ROS2 TurtleBot3 Burger
- **障碍如何进入学习**: **稀疏终端**：到目标 +100；`min(LIDAR)<0.2m` 或超时 −10；中间步 0——障碍只经观测几何与稀疏碰撞罚进入，**无显式 obstacle cost 头**
- **与我们卡点的关系**:
  - 本批里 **唯一直接验证「WM + imagination 做无教师导航/避障」** 的栈，叙事上最支撑我们方法族
  - 但传感是 **几何 LIDAR**，不是视觉 RGB 世界模型；障碍在 latent 里是隐式的，没有我们要写的 **action-conditioned obstacle cost**
  - 稀疏碰撞与我们「学不动 OA / 贴墙」同类风险——他们靠 WM 样本效率 + 全向测距缓解，未解 urban 视觉死胡同
  - 无 shield 训练–部署分离
- **可抄实现点**:
  - 高维传感先 VAE/MLP 再进 RSSM（我们已有视觉 encoder，可作「几何旁路 ablation」对照）
  - Imagination 训策略 vs model-free 的 **同任务对照表** 写法（论文表格口径）
  - 难关 curriculum：先密小场再长大场，拼预训练 warm-start
  - 固定控制步频，避免训/评墙钟不一致（部署相关）
- **局限/不适配**: 地面车 + LIDAR，非 aerial monocular；稀疏奖励无 clearance shaping；场景是小房间障碍场，非 urban interior 长程；无显式风险/盾
- **推荐优先级**: **高**（WM/imagination 无教师导航的直接先例；写 related work / 差异点必引；实现上可抄对照实验结构多于抄传感）

开源: https://github.com/raulsteinmetz/turtlebot-dreamerv3

---

### GRaD-Nav — GRaD-Nav: Efficiently Learning Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics
- **出处**: Chen, Sun, Gao, Low, Chen, Schwager (Stanford); arXiv:2503.03984v3, 2025
- **无教师？**: **部分**（相对 SOUS VIDE 等 **明确不做 IL / 不要专家飞行数据**；但是用 3DGS 点云 **预规划 waypoint + 参考轨迹** 做稠密 shaping，且 **critic 有特权位姿/深度**——非纯视觉无特权）
- **方法摘要**:
  - 混合仿真：可微四旋翼动力学（body rate + 归一化推力）+ **3DGS** 渲第一人称 RGB；点云兼作碰撞检查与参考轨迹
  - 算法：SHAC 风格短地平线可微 actor-critic（GRaD-Nav）；SqueezeNet 视觉嵌入 + **CENet（β-VAE）** 估环境上下文 latent，供策略运行时适应
  - 奖励：生存/姿态/高度/动作平滑 + yaw 对齐 + waypoint + **FOV 内最近障距** + 轨迹跟踪；域随机化质量/推力/惯量/电机延迟
  - Curriculum：多 3DGS 场景轮训（门位置不同）→ in-task 泛化；零样本到 Orin + RealSense 真机穿门
- **传感/动作**: 训练/部署 **RGB**；观测含高度、姿态、速度、当前/上步动作（无显式 xy）；动作 = 期望 body rate + 推力；critic 额外用位置 + 下采样深度
- **障碍如何进入学习**: 仿真点云碰撞检查 + 奖励项 \(r_{\mathrm{obst}}=d_{\mathrm{obst}}\)（FOV 最近障，阈值内）；几何梯度经可微动力学回传——**非 demo**，但是 **手写轨迹引导 + 特权价值**
- **与我们卡点的关系**:
  - 视觉无人机 + 无 IL 学穿障，表面贴近；实际是 **可微仿真 BPTT**，不是 Dreamer imagination / latent obstacle cost
  - CENet 对「运行时环境变体」有用，但对我们「关罩训贴墙、开罩兜底」不对口
  - 依赖场景级 3DGS 重建与参考轨迹——urban AirSim 可程序化，但「每个场景先重建再训」成本与我们交互学 OA 目标不同
  - 特权 critic + waypoint 引导削弱「无教师」纯度；作者自承 reward/轨迹手工限制多任务
- **可抄实现点**:
  - CENet / 历史观测 β-VAE 作 runtime context（可贴在策略前，作 antistick/场景适应 ablation）
  - 短地平线可微 AC vs PPO 的样本效率对照叙事
  - FOV 最近障距作为稠密 clearance 项（对照我们 obstacle cost / NavRL log-ray）
  - 多场景 curriculum 重置学习率做 in-task 泛化
- **局限/不适配**: 门/短程轨迹任务，非长程 urban goal；需 3DGS + 可微动力学栈；无 WM imagination；无 shield ON/OFF；依赖可靠速度估计（VIO）
- **推荐优先级**: **中**（视觉无人机无 IL 的工程参考；与我们栈差异大，优先抄 CENet/clearance 项而非整栈）

开源: https://github.com/Qianzhong-Chen/grad_nav  
附注: `txt/grad_nav.html` 为同文 arXiv HTML 版，无额外笔记内容。

---

### NPE-DRL — （文件名待替换）
- **出处文件**: `txt/npe_drl.txt` / 对应 pdf **下错论文**
- **实际内容**: *Friction-Aware Safety Locomotion for Wheeled-legged Robots using Vision Language Models and Reinforcement Learning*（轮腿/WIP + VLM 估摩擦系数 + RL）——**与无教师视觉 OA / 航拍导航无关**
- **处理**: **不分析**；标记 **待替换**。应按原意图重新下载真正的 NPE-DRL（或原清单中的同名 OA 论文）后再补笔记。

---

### SIGN — （本批跳过）
已收录于 [batch-B.md](./batch-B.md)：训练不管死安全、部署用 action-conditioned \(Q_c\) 纠动作——与关罩训/开罩部署同构。此处不重复。

---

## 本批结论

按「无教师主动学 OA + 关罩训/开罩部署 + latent/想象成本」对我们的可用度排序：

| 排名 | 短名 | 优先级 | 一句话用途 |
|------|------|--------|------------|
| 1 | **MonoMPC** | 高 | \(o{+}u\)→clearance 分布 + 下游风险联合校准；最贴 action-conditioned obstacle cost（但非无教师 RL） |
| 2 | **TurtleBot-DreamerV3** | 高 | Dreamer imagination 无教师导航的直接先例；用来钉「我们 = 视觉 WM + 显式障碍成本」差异 |
| 3 | **GRaD-Nav** | 中 | 视觉无人机无 IL + FOV clearance / CENet；栈是可微仿真，非 latent imagination |
| — | **NPE-DRL 文件** | — | **待替换**（当前是轮腿摩擦 VLM，勿引用） |
| — | **SIGN** | — | 已在 Batch B |

**本批最该立刻抄进实验/写作的三件事**：
1. **MonoMPC 式代价形态**：学 worst-case clearance（或 risk）分布而非只学二值撞；用下游风险头校准 \(\sigma\)——直接服务我们 latent obstacle cost 设计与校准指标（UR/CR/ECE）。
2. **TurtleBot-DreamerV3 对照叙事**：同任务上 WM+imagination vs model-free；我们补一句「他们靠 LIDAR 几何隐式避障，我们在视觉 latent 上挂显式 action-conditioned cost」。
3. **GRaD-Nav 的 FOV \(d_{\mathrm{obst}}\) / CENet**：作稠密 antistick 与运行时上下文的轻量 ablation，不必上 3DGS 全栈。

**对本批与前两批合起来的差异点**：MonoMPC 证明「动作条件障碍模型 + 规划打分」可行，但监督来自特权测距；TurtleBot-Dreamer 证明 imagination 能无教师导航，但无显式障碍成本且非视觉。**仍缺一篇把「视觉 RSSM imagination + action-conditioned obstacle cost + 关罩训」三条同时做满的先例**——这正是 Aerial WAM v2 应坚持写清的贡献缝。
