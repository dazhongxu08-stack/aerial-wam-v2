# Batch A — 无教师避障相关论文笔记

目标对照（Aerial WAM v2）：视觉世界模型 + 想象规划，**主动学避障、无 PathExpert/BC**；动作条件障碍代价头；**关罩训 / 开罩验**；AirSim 城市硬路线。

---

### ProbColl — Uncertainty-Aware Reinforcement Learning for Collision Avoidance
- **出处**: arXiv:1702.01182 (BAIR / Kahn, Villaflor, Pong, Abbeel, Levine), 2017
- **无教师？**: **是** — 无专家示教、无模仿；碰撞标签来自自车碰撞指示（需体验碰撞）；有不确定性估计，但无地图/特权几何 GT
- **方法摘要**:
  - 学判别式碰撞预测网 \(P_\theta(\mathrm{COLL}\mid x_t,u_{t:t+H},o_t)\)：当前观测 + **未来动作序列** → 地平线内是否撞
  - Bootstrap + dropout 估计模型不确定性，构成 risk-averse 碰撞概率 \(\tilde P = \sigma(\mathbb{E}[f]+\lambda_{\mathrm{STD}}\sqrt{\mathrm{Var}[f]})\)（在 sigmoid 前激活上加方差）
  - 速度相关碰撞代价 \(C_{\mathrm{COLL}}=\lambda_{\mathrm{COLL}}\|v\|^2\)，与 \(\tilde P\) 相乘进入 MPC 代价 → 不确定时自动减速，只经历低速碰撞
  - 模型基 RL：用当前预测模型采样/规划 → 聚合轨迹 → 再训碰撞头（Alg. 2）
  - 仿真/真机四旋翼 + RC 车；MPC 在固定动作序列库上取最低代价
- **传感/动作**: 16×16 灰度图（仿真/真机四旋翼）；\(u=\)平面速度指令；MPC 约 190 条直线恒速轨迹
- **障碍如何进入学习**: **预测碰撞概率作规划代价**（非稀疏 reward shaping 为主）；标签=是否撞；不确定性调节探索速度
- **与我们卡点的关系**:
  - **动作条件代价**：几乎同构（obs+action seq → collision cost）
  - **关罩训/开罩验**：训时用不确定+速度代价当“软罩”；可借鉴“训时保守、熟了加速”，非硬开关罩
  - **城市楼群**：未做；场景多为单障碍/走廊级
  - **想象规划**：短地平线 MPC 扫动作序列，可直接接到我们想象 planner 的候选轨迹打分
- **可抄实现点**:
  - 在 latent 上挂 Bernoulli 碰撞头，输入拼接/RNN 吃 \(u_{t:t+H}\)
  - 不确定性用 **pre-sigmoid 方差 + \(\lambda_{\mathrm{STD}}\)**，不要对概率直接加常数保守偏置
  - \(C = C_{\mathrm{task}} + \tilde P\cdot\lambda\|v\|^2\)；离线切片长度 \(H\) 的 (o,u seq, collide∈[t,t+H]) 交叉熵
  - Bootstrap 条数：仿真可多（文中 50），在线可少（真机 5）
- **局限/不适配**:
  - 必须撞（虽可控速）；无世界模型滚动想象，仅判别碰撞
  - 观测极低分辨率；任务非城市峡谷
  - 动力学假定已知，只学代价
- **推荐优先级**: **高**

---

### BADGR — BADGR: An Autonomous Self-Supervised Learning-Based Navigation System
- **出处**: arXiv:2002.05700 (BAIR / Kahn, Abbeel, Levine), 2020
- **无教师？**: **是** — 无仿真、无人工标注、无专家演示；碰撞/颠簸/位姿由机载传感器事后自标；收集策略为时间相关随机游走（偶需人重置翻车）
- **方法摘要**:
  - Off-policy 大规模自采：Jackal 上 RGB+IMU/GPS/轮速等
  - 自监督事件标签：碰撞（城市场景用激光近距 / 越野用 IMU 卡住）、bumpiness（角速度阈值）、position（里程计融合）
  - CNN→LSTM：**图像 + 未来动作序列** → 预测未来 \(H\) 步各事件
  - 部署时人手写 reward（到目标 − 碰撞 − 颠簸），对预测事件做 MPC；随机优化器用时间相关采样 + reward-weighted 更新动作序列
  - 约 42h 真机数据；可泛化新环境并继续自采改进；可走过高草等“几何假障碍”
- **传感/动作**: 双 170° RGB；动作=线/角速度；规划 horizon \(H\) 步
- **障碍如何进入学习**: **预测多任务事件**，再经用户定义 reward 进规划；碰撞是自标二值事件，非特权地图
- **与我们卡点的关系**:
  - **动作条件代价**：核心同构（action-conditioned predictive model）
  - **关罩训/开罩验**：训数据用随机政策（“无罩乱开”）→ 训预测模型 → 部署用 reward 规划（“软罩”）；很贴近关罩采/开罩用
  - **城市楼群**：有城市场景，但是地面车，非楼群峡谷飞
  - **想象规划**：显式未来事件预测 + 采样规划，可映射到我们想象 rollouts 上读 cost head
- **可抄实现点**:
  - 一条 LSTM 并行预测 collision / progress / 其它辅助事件；CE+MSE 混合损失
  - 时间相关随机采数（避免纯均匀随机导致几乎只直行）
  - 规划：warm-start 的 reward-weighted 随机优化（Nagabandi 式），不要只 random shooting
  - 撞后自动后退旋转 reset；城市用测距/深度近距作撞标，减少破坏
- **局限/不适配**:
  - 地面车；需真实碰撞/卡住体验；无视觉世界模型
  - 目标依赖 GPS 稀疏航点；飞行/3D 楼群未覆盖
  - 城市场景仍用 LIDAR 作碰撞检测特权传感（标签侧）
- **推荐优先级**: **高**

---

### Collision Replay — Collision Replay: What Does Bumping Into Things Tell You About Scene Geometry?
- **出处**: arXiv:2105.01061 (Raistrick, Kulkarni, Fouhey / Michigan), 2021
- **无教师？**: **是（几何侧）** — 无专家、无 GT 深度/floorplan 作主监督；监督=随机游走碰撞（或近距）回放；评测对照强监督时有特权 GT，但方法本身不依赖
- **方法摘要**:
  - 核心：**Collision Replay** — 撞点回标更早时刻：\(k\) 步前观测 → “还有 \(k\) 步会撞”的样本
  - 学 **航向条件** 的碰撞时间分布 \(P_T(t\mid x,\alpha)\)（多项分类到 \(0..k\) 或回归期望）
  - 分布可转距离函数 / 近似 floorplan（最短有支撑的时间 ≈ 到障距离）
  - PIFu 式隐式：图像特征 + 地面点 + 相对航向 → 分布；也做 egocentric RGB/声呐
  - Habitat + Gibson/Matterport；有执行噪声；随机政策训、测时不必随机
- **传感/动作**: RGB（及可选 binaural spectrogram）；动作主要是前走/原地转；随机游走采数
- **障碍如何进入学习**: **碰撞作稀疏自监督 → 学 TTC/距离场**；不是直接控策略的 reward；规划若用需另接
- **与我们卡点的关系**:
  - **动作条件代价**：弱相关——有航向/下一步动作条件，但输出是几何 TTC 分布，非动作序列代价
  - **关罩训/开罩验**：无此设定
  - **城市楼群**：室内仿真，非城市外景楼群
  - **想象规划**：可作 cost 的几何先验/稠密化标签源，而非 planner 本体
- **可抄实现点**:
  - **碰撞回放稠密化**：撞一次 → 标过去 \(H\) 帧的剩余步数/“会撞”标签（我们现有 horizon 碰撞头可直接用）
  - 预测分布而非单点概率；用 CDF 阈值估最短距离，减轻随机政策期望 TTC 偏大
  - 近距传感器代替真撞（`<10cm`）以保护机体
  - 中途撞时把后续步标签全置 1，并对撞后未执行动作做随机替换增广（与 ORACLE/BADGR 系一致）
- **局限/不适配**:
  - 目标是场景几何，不是闭环避障控制；无想象 WM
  - 随机游走标签噪声大；飞行 3D/城市硬线未验证
  - 远程点预测依赖地面投影假设
- **推荐优先级**: **中**

---

### Future Depth — Future Depth as Value Signal for Learning Collision Avoidance
- **出处**: 会议稿风格 / KU Leuven（Kelchtermans & Tuytelaars）；项目页 depth-q-net；对照 Kahn GCG [12]
- **无教师？**: **部分** — 无专家示教、无 BC；Depth-Q-Net **不需要真实碰撞**；但训练标签用 **LiDAR 未来深度**（特权测距）；声称可换成单目深度估计网使标签侧也无额外传感器
- **方法摘要**:
  - 论证：把深度当累计 return 会数学上推到“靠近障碍更好”；改用 **myopic**：\(Q(I_t,a_t)=D_{t+1}\)
  - Depth-Q-Net：MobileNet 特征 ⊕ 动作 → 预测下一帧 LiDAR 深度 bin；策略 \(\arg\max_a \min(\hat D_{t+1})\)（让最近障尽量远），深度 clip 到 2m
  - Baseline Coll-Q-Net：同结构预测地平线内碰撞概率（需撞标），\(\arg\min_a Q\)
  - Off-policy：OU 过程随机 yaw 采数；冻住 ImageNet 特征更稳
  - Gazebo 峡谷 Turtlebot：同数据量下 Depth-Q 更稳、更少数据即可通，且无需撞
- **传感/动作**: 单目 RGB；动作=连续 yaw（测时量化 -1/0/1）；固定线速度；标签侧 LiDAR
- **障碍如何进入学习**: **未来深度作稠密 action-value**（预测代价/价值）；碰撞基线才用稀疏撞标
- **与我们卡点的关系**:
  - **动作条件代价**：强相关——\(Q(o,a)\) 直接是未来可通行性代理
  - **关罩训/开罩验**：未讨论；但“用稠密深度信号代替撞”可减轻关罩乱撞的破坏/稀疏性
  - **城市楼群**：仅仿真峡谷，无楼群
  - **想象规划**：单步 myopic；可把“预测未来深度/ clearance”接到多步想象的代价累加
- **可抄实现点**:
  - 在 latent 上加 **action-conditioned future clearance / min-depth** 头，用下一帧深度（或 WM 解码深度）作监督，缓解碰撞标签稀疏
  - 策略规则：对候选动作批前向，取 \(\max_a \min_i \hat d_i\)（或对想象轨迹积分 clearance）
  - 对比实验模板：同 backbone 的 Coll-Q vs Depth-Q，看数据量–成功率曲线
  - 深度标签可噪声（估计深度亦可），损失用 L1
- **局限/不适配**:
  - 现实现仍依赖 LiDAR GT；单步、无目标导航
  - 无不确定性、无世界模型、非无人机实验
  - 峡谷场景过简，迁到城市楼群 clearance 定义更难（侧面/上方障碍）
- **推荐优先级**: **中高**

---

### Self-Learning CAS — Learning Through Experience: A Self-Learning Collision Avoidance System
- **出处**: 硕士论文，GIST / Hyunjun Han，2018（导师 Heung-No Lee）
- **无教师？**: **是** — 无预采集专家集、无模仿；在线试错自建动态数据集；但标签来自 **仿真格子合法性**（出界/撞障=collision），属特权状态判定
- **方法摘要**:
  - 单前向距离传感器序列 + 速度 → LSTM 二分类（撞 / 不撞）
  - **Adverse-event 学习**：采数与训练交替；发现新碰撞序列 → 用全部记忆全批训 LSTM → reset
  - 未见过的序列先硬前进以探索；已见过的序列：预测为撞则旋转，否则随机动
  - 25×25 网格、随机方块障碍；动作前走/±45°转；距离最大 20 格
  - 报告分类准确率 ≥99.25%（乃至 ~100%）；另用手工转向规则做到达测试
- **传感/动作**: 1D 前向距离 + 自推速度；离散网格动作
- **障碍如何进入学习**: 撞/非法状态 → 监督标签；部署时 LSTM 作分类器触发转向（非 reward、非预测规划）
- **与我们卡点的关系**:
  - **动作条件代价**：很弱——输入几乎无显式候选动作，是状态序列分类
  - **关罩训/开罩验**：无；训时直接撞格子
  - **城市楼群 / 想象规划**：无
- **可抄实现点**:
  - “新碰撞样本触发一轮集中更新再继续探索”的在线课程节奏（若我们做持续自采）
  - 撞序列记忆去重，避免重复无效探索
- **局限/不适配**:
  - 玩具 2D；无视觉/WM；到达依赖手工规则 + 坐标特权
  - 分类准确率≠闭环难环境避障；与 WAM 管线差距过大
- **推荐优先级**: **低**

---

### ORACLE — Motion Primitives-based Navigation Planning using Deep Collision Prediction
- **出处**: arXiv:2201.03254 (Nguyen, Fyhn, De Petris, Alexis / NTNU), 2022；系统名 ORACLE
- **无教师？**: **是（控制侧）** — 无 BC/专家轨迹；碰撞预测网在 **仿真随机动作序列** 上自标碰撞；部署无地图、无位置，仅深度+速度；有全局规划器给目标航向（拓扑级，非示教飞）
- **方法摘要**:
  - 固定 **速度–转向角** 运动基元库（轨迹保证在深度 FOV 内）
  - CPN：滤波深度图 + 部分状态 \((v,\omega)\) → LSTM 吃动作序列 → 逐步碰撞概率；加权 BCE（正类加权提 recall）
  - 不确定：状态用 **Unscented Transform** 传播；模型用 **MC dropout**；代价 \(\hat c_{\mathrm{uac}}=\bar\mu+\alpha\sqrt{\sigma_{\mathrm{tot}}}\)
  - 规划：先滤掉代价 > \(\hat c_{\min}+c_{\mathrm{th}}\) 的基元，再在剩余中选最贴近全局目标航向者；RHC 执行首步
  - 撞中途：后续标签置 1 + 撞后动作随机替换增广 + 水平翻转增广；~1.5M 点；走廊与矿井真机
- **传感/动作**: RealSense 深度；状态仅线/角速度（无位置）；动作=\((v^r,\psi^r)\) 基元
- **障碍如何进入学习**: **动作序列条件碰撞预测 → 规划代价/过滤**；全局目标只作次级代价
- **与我们卡点的关系**:
  - **动作条件代价**：高度对齐（深度/特征 + action seq → collision cost）
  - **关罩训/开罩验**：训时随机飞收集撞标（类关罩），部署用代价阈值过滤（类开罩）；可直接借鉴“训采随机 / 部署硬阈值”
  - **城市楼群**：室内走廊+矿井窄道，非室外楼群，但杂乱/感知退化贴近“硬环境”
  - **想象规划**：MPL 并行打分 ≈ 我们候选想象轨迹打分；可把 CPN 换成 WM latent 上的 cost head
- **可抄实现点**:
  - 网络拆 CNN / Combiner / Prediction，MC dropout 只打 CNN，批维 \(N_{\mathrm{MC}}\times N_\Sigma\times N_{\mathrm{MPL}}\) 并行
  - 碰撞代价时间衰减加权 \(\sum_i \hat c_{t+i}e^{-\lambda(i-1)}\)（早撞更重）
  - 两阶段：先安全过滤再追目标，避免 goal reward 压过避障
  - 标签增广：撞后动作随机替换 + 水平翻转；正类加权
  - 深度 IP-Basic 等滤波缩小 sim2real 缺口
- **局限/不适配**:
  - 无学习世界模型；依赖深度（非纯 RGB WM）
  - 仿真动力学需与真机对齐；全局航向仍外置
  - 不确定估计算力重（需 NX 级）
- **推荐优先级**: **高**

---

## 本批结论

对本项目（无教师、动作条件障碍代价、关罩训/开罩验、想象规划、城市硬线）的优先级排序：

1. **ORACLE** — 架构最可直接抄：MPL/候选轨迹上的动作序列碰撞头 + 不确定代价 +「先滤障再追目标」；训随机采、部署阈值罩，对应关罩训/开罩验。
2. **ProbColl** — 理论与代价形式最干净：\(\tilde P(o,u_{1:H})\cdot\|v\|^2\) + bootstrap/dropout；适合接到我们 latent cost head 与想象打分。
3. **BADGR** — 工程完备的自监督动作条件多事件预测 + 采样 MPC；关罩乱采→开罩规划的系统故事最完整（地面/RGB，需改飞与 3D）。
4. **Future Depth** — 用稠密未来 clearance 代替稀疏撞标，专治“无教师又不敢狂撞 / 标签太稀”；可与碰撞头并列作辅助。
5. **Collision Replay** — 作 **标签增广/TTC 几何先验**，不单独当主规划器。
6. **Self-Learning CAS** — 概念（adverse online）有启发，实现维度过低，优先级最低。

**本批最该落地的组合**：ORACLE/ProbColl 式动作序列碰撞头（+不确定）作主代价；BADGR 式 off-policy 自采与规划接口；Future Depth / Collision Replay 作稠密辅助监督，减轻纯碰撞稀疏与真撞依赖。
