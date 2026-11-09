# 后果模型的设计依据：阅读清单

按推荐阅读顺序编号。前 4 篇是主干，读完就能看懂现在的后果模型和规划器；5–7 篇对应辅助信号和已知缺口；8–9 篇只和"关罩训练、关罩报数"这条口径有关。

详细版（逐篇原文摘录、页码出处、我们的说法与论文实际支持程度对照）见 `后果模型设计依据-论文摘录.md`。

原始 PDF 在 `../pdfs/`，抽取的全文在 `../txt/`，逐篇分析笔记在 `../notes/batch-A.md`（1–6）、`../notes/batch-B.md`（7–9）和 `../notes/batch-C.md`（MonoMPC）。

| # | 文件 | 论文 | 页数 | 对应我们的设计 |
|---|---|---|---|---|
| 1 | `01_BADGR.pdf` | Kahn, Abbeel, Levine. *BADGR: An Autonomous Self-Supervised Learning-Based Navigation System*. RA-L 2021, [arXiv:2002.05700](https://arxiv.org/abs/2002.05700) | 10 | 整体框架和网络结构：卷积编码画面作为初始状态，循环网络沿候选动作序列逐步预测事件；随机策略采数据，事后自动标注；每步重新规划 |
| 2 | `02_ProbColl.pdf` | Kahn, Villaflor, Pong, Abbeel, Levine. *Uncertainty-Aware Reinforcement Learning for Collision Avoidance*. [arXiv:1702.01182](https://arxiv.org/abs/1702.01182) | 12 | 给定画面和动作序列预测碰撞概率；用模型集成估计不确定性（我们的 3 种子平均） |
| 3 | `03_ORACLE.pdf` | Nguyen, Fyhn, De Petris, Alexis. *Motion Primitives-based Navigation Planning using Deep Collision Prediction*. ICRA 2022, [arXiv:2201.03254](https://arxiv.org/abs/2201.03254) | 8 | 固定的动作基元库（我们的 11 种候选）；先按预测碰撞风险过滤，再在安全的基元里选最靠近目标的（我们的"生存率 ≥ 0.95 后选进展最大"） |
| 4 | `04_MonoMPC.pdf` | *MonoMPC: Monocular Vision Based Navigation with Learned Collision Model and Risk-Aware Model Predictive Control*. RA-L 2025, [arXiv:2508.07387](https://arxiv.org/abs/2508.07387) | 9 | 单目估计的深度太噪，不能当障碍物地图，应该把它作为条件，学习和动作相关的碰撞风险（我们用 DA3 预测深度作输入，不用真实深度） |
| 5 | `05_FutureDepth.pdf` | Kelchtermans, Tuytelaars. *Future Depth as Value Signal for Learning Collision Avoidance* | 6 | 用未来深度作稠密训练信号（我们的前方、左右净空输出）；也说明了把深度混进累计回报会把"贴着障碍"算成有价值，这是我们不把安全和进展加成一个分数的理由之一 |
| 6 | `06_CollisionReplay.pdf` | Raistrick, Kulkarni, Fouhey. *Collision Replay: What Does Bumping Into Things Tell You About Scene Geometry?* [arXiv:2105.01061](https://arxiv.org/abs/2105.01061) | 17 | 学"还有多久会撞"的分布，和我们每步碰撞风险连乘成生存率的做法相近 |
| 7 | `07_MAVRL.pdf` | Yu, De Wagter, de Croon. *MAVRL: Learn to Fly in Cluttered Environments with Varying Speed*. [arXiv:2402.08381](https://arxiv.org/abs/2402.08381) | 8 | 带记忆的隐状态。这是我们现在缺的：v8 只看当前一帧，楼一移出画面就"看不见"了 |
| 8 | `08_SIGN.pdf` | Yan et al. *SIGN: Safety-Aware Image-Goal Navigation for Autonomous Drones via Reinforcement Learning*. [arXiv:2508.12394](https://arxiv.org/abs/2508.12394) | 8 | 训练时不挂安全罩，安全模块只在部署时另加；主结果要报关罩 |
| 9 | `09_NavRL.pdf` | Xu, Han, Shen, Jin, Shimada. *NavRL: Learning Safe Flight in Dynamic Environments*. RA-L 2025, [arXiv:2409.15634](https://arxiv.org/abs/2409.15634) | 8 | 同上，训练和安全模块分开 |

## 每篇重点读哪几节

**1. BADGR**：III-B 数据采集、III-C 自监督标注、III-D 预测模型、III-E 规划。III-D 的网络图和我们的结构几乎一一对应，III-E 是"采样一批动作序列、打分、执行第一步"的规划方式。

**2. ProbColl**：IV-A 带不确定性的碰撞预测、IV-C 网络结构、IV-D 用神经网络估计不确定性（集成和 dropout）、V-A 四旋翼实验。

**3. ORACLE**：IV-A 动作基元库、IV-C 带不确定性的预测、IV-D 导航策略（先过滤再选目标方向）。

**4. MonoMPC**：III-A 把视觉导航写成带风险的轨迹优化、III-D 和 III-E 两种学碰撞模型的方式（基线做法和面向下游任务的做法）、IV-B 学到的碰撞模型的验证。

**5. Future Depth**：III 方法。篇幅短，可以全读。

**6. Collision Replay**：3 方法，附录 F 和 "Learning to Fly by Crashing" 的对比。

**7. MAVRL**：III-A 深度图编码、III-B 带记忆的隐状态表示、V-A 隐状态实验。

**8. SIGN**：IV 安全模块（IV-A 训练碰撞概率预测、IV-B 动作纠正）。

**9. NavRL**：看安全模块和训练分开的部分即可。

## 哪些不是来自论文

- **分支采样**：从同一状态出发，把每种动作都在模拟器里实际飞一遍。依据是我们自己的 G0 诊断：原来的世界模型分不清"小侧移"和"离开墙"。
- **从失败状态出发的分支采样**（v9、v10）：依据是我们自己的校准检查，碰撞前状态上模型过度自信。
- **高度规则、可见性约束**：项目约定，不是论文。
- **推演 10 步、生存率门槛 0.95、网络大小**：经验值，门槛是在训练路线上比较后定的。
