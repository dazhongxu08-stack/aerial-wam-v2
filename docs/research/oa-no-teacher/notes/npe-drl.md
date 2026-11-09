# NPE-DRL（对照：有弱教师，非无教师主线）

- **出处**: Zhang, Yan, Xiao, Feroskhan — IEEE TAI 2024/2025；DOI [10.1109/TAI.2024.3464510](https://doi.org/10.1109/tai.2024.3464510)（**无公开 arXiv PDF**）
- **代码**: https://github.com/zzzzzyh111/NonExpert-Guided-Visual-UAV-Navigation-Gazebo
- **无教师？**: **否** — 用 **APF（人工势场）** 作为 non-expert 在线示教，再渐进切到 D3QN 自主探索；不是专家演示，但仍是**规则教师**
- **方法摘要**:
  - Gazebo 视觉受限 UAV 避障；Dueling Double DQN
  - 训练前期模仿 APF 动作，后期加大 RL 探索比例
  - 目标：提样本效率、密障场景适应性；有仿真+真机
- **与我们关系**: 若坚持「无教师」，NPE **不作为主抄**；可作消融：「弱规则教师 vs 纯自监督代价头」对照
- **推荐优先级**: **低（主线）/ 中（消融对照）**
