# Related Work (draft for LaTeX)

Use this section as the backbone for **§2 Related Work** or split into subsections in the main paper. Citations use placeholder keys — import from `references.bib` or the seed at the end of `~/Desktop/aerial-wam-v2-paper-outline.md`.

---

## 2.1 World-model-based navigation

**Navigation World Models (NWM)**~\cite{bar2024nwm} train a large conditional diffusion transformer on egocentric video and actions, then plan by imagining future frames and ranking trajectories (MPC or policy proposals). The model stays in **pixel/latent video space** at planning time and scales to diverse embodiments, but does not target **onboard aerial deployment** under a strict monocular contract.

**Vid2World**~\cite{zhang2025vid2world} distills pretrained **video diffusion** into action-conditioned interactive world models. The distillation target is still a **generative video model**, not a compact **Dreamer-style RSSM** suitable for 5–10\,Hz closed-loop control on edge hardware.

**MUN**~\cite{mun2024} learns **unconstrained transitions between subgoals** in replay for long-horizon goal navigation on legged agents. It shares our use of **milestones/subgoals** but does not address **aerial dynamics**, **monocular exteroception only**, or **runtime safety shields**.

**DreamerNav**~\cite{dreamernav2025} extends DreamerV3 with **depth images plus a structured occupancy map** and hybrid global–local planning for **indoor quadrupeds**. It demonstrates world-model navigation on real robots but **relies on depth and explicit maps**, unlike our deploy-time **RGB-only exteroception**.

**Our distinction:** offline **Wan-class video pretraining** is distilled into a **fast TorchRSSM** that runs online with **imagination planning**; we do not run a billion-parameter diffusion planner onboard.

---

## 2.2 Model-based RL for aerial agents

**Dream to Fly**~\cite{dreamtofly2025} and **SkyDreamer**~\cite{skydreamer2025} show that **DreamerV3** can map **pixels to agile quadrotor commands** in **racing** domains, including sim-to-real transfer at high speed. These works establish the viability of **latent world models for UAV control**, but the task is **gate racing**, not **tens-to-hundreds of meters goal-first navigation** in urban layouts with **unknown goal locations**.

**AirDreamer**~\cite{airdremer2026} is the closest aerial counterpart: a **DreamerV3 RSSM** plus sparse-reward policy for **generalist drone navigation** in cluttered scenes, with emergent yaw and detours. Critical differences for our paper:

| Dimension | AirDreamer | Aerial WAM v2 (ours) |
|-----------|------------|----------------------|
| Exteroception at deploy | Uses **depth** in the multimodal encoder | **Monocular RGB only**; depth is **predicted** ($\hat{D}$), not sensed |
| Teacher / pretrain | Standard Dreamer training pipeline | **Wan2.2 video–action joint model** distilled to RSSM (**offline only**) |
| Planning stack | WM imagination + RL policy | **AdaptiveSubgoal** (20–55\,m) + **ImaginationPlanner** ($H{=}5$) + **LatentActor** |
| Safety | Learned policy + sparse reward | **Three-zone shield** on $\hat{D} \cup \tau \cup p_{\mathrm{coll}}$ with audited latch behavior |
| Task benchmark | General navigation maps | **16-route urban outdoor** acceptance + Step-G **153\,m** routes |
| Verification | Paper metrics | **Frozen gate protocol** (V0–V4 signals, primary/secondary criteria) |
| Real platform | Real drone demos reported | **Orin + Pixhawk + USB camera** deploy stack (flight table in progress) |

**NavRL**~\cite{navrl2024} combines PPO with a **velocity-obstacle-inspired safety shield** for dynamic scenes. It is **model-free**, not a world-model approach, and targets **dynamic obstacles** rather than our **long-range static urban** setting.

---

## 2.3 Monocular vision navigation (non–world-model)

**MonoMPC**~\cite{monompc2025} uses monocular RGB with a **learned collision distribution** and risk-aware MPC, explicitly avoiding noisy zero-shot depth for collision checking. **Invisible Servoing**~\cite{invisibleservoing2024} plans in latent space with diffusion to reach a **target view** on multirotors. **ViNT / NoMaD / GNM** (imitation topological navigation) excel at **traversability** over long horizons but typically **do not maintain a recurrent latent dynamics model** for multi-step imagination under our sensing contract.

These lines justify **Table~\ref{tab:related}**: monocular methods often lack **learned long-horizon dynamics**; world-model UAV papers often use **depth or racing tasks**.

---

## 2.4 Semantic / VLM outdoor aerial navigation

**RAVEN**~\cite{raven2025} and **CityNavAgent**-style systems use **semantic maps, rays, or LLM landmarks** for **outdoor aerial search**. They are complementary: we keep the **WAM control core** coordinate-driven; semantic goals are an **optional adapter** (vgoal track), not the Phase-2/3 mainline.

---

## LaTeX Table 1 — Positioning (copy into paper)

```latex
\begin{table*}[t]
\centering
\caption{Comparison with closely related navigation systems. \textbf{Ours} denotes Aerial WAM v2. Deploy exteroception: sensors available at runtime for obstacle/goal reasoning (proprio/IMU allowed for all).}
\label{tab:related}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
Method & WM family & Deploy exteroception & Platform & Task & Long-range ($\gtrsim$50\,m) & Safety layer \\
\midrule
NWM~\cite{bar2024nwm}           & Video DiT      & RGB            & Ground / human & General nav      & \checkmark & Planning-time only \\
Vid2World~\cite{zhang2025vid2world} & Video DiT  & RGB            & Sim robots     & Manip / nav      & \checkmark & None \\
MUN~\cite{mun2024}              & Dreamer RSSM   & State / sim    & Legged         & Subgoal nav      & \checkmark & None \\
DreamerNav~\cite{dreamernav2025}& Dreamer RSSM   & Depth + map    & Quadruped      & Indoor dynamic   & \texttimes  & Implicit \\
Dream to Fly~\cite{dreamtofly2025}& Dreamer RSSM & RGB pixels   & Quadrotor      & \textbf{Racing}  & \texttimes  & None \\
SkyDreamer~\cite{skydreamer2025}& Informed Dreamer& RGB pixels    & Quadrotor      & \textbf{Racing}  & \texttimes  & None \\
AirDreamer~\cite{airdremer2026} & Dreamer RSSM   & \textbf{Depth} & Quadrotor      & General nav      & \checkmark & Learned only \\
MonoMPC~\cite{monompc2025}      & None (MPC)     & RGB            & Ground / aerial& Cluttered nav    & \checkmark & Risk MPC \\
NavRL~\cite{navrl2024}          & None (PPO)     & Range / state  & Quadrotor      & Dynamic obstacles& Medium   & VO shield \\
ViNT / NoMaD                    & None (IL)      & RGB            & Ground         & Topological nav  & \checkmark & None \\
\midrule
\textbf{Ours (Aerial WAM v2)}   & \textbf{Distilled RSSM} & \textbf{RGB only} & \textbf{Quadrotor} & \textbf{Goal-first urban} & \checkmark & \textbf{Three-zone shield} \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## LaTeX Table 2 — Capability matrix (optional appendix)

```latex
\begin{table}[t]
\centering
\caption{Capability checklist (filled from project acceptance logs). Real-flight rows marked \textit{TBD}.}
\label{tab:capabilities}
\small
\begin{tabular}{lcc}
\toprule
Capability & Simulation (Phase-2) & Real Orin deploy \\
\midrule
Unknown goal, partial observability        & \checkmark & \textit{TBD} \\
Monocular RGB exteroception (no SLAM map)  & \checkmark & \checkmark \\
Predicted metric depth $\hat{D}$ (DA3 head) & \checkmark & \checkmark \\
FOE time-to-contact $\tau$ channel         & \checkmark & \checkmark \\
Imagination planning ($H{=}5$)           & \checkmark & \checkmark \\
Hierarchical subgoals (20--55\,m)        & \checkmark & \textit{TBD} \\
16-route urban benchmark                   & 86.7\% SR  & --- \\
Max demonstrated route length            & 153.7\,m (Step G) & \textit{TBD} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Suggested paragraph to introduce Table~\ref{tab:related}

> Recent work explores world models for navigation at increasing scale and embodiment diversity. However, **no published system simultaneously (i) enforces monocular RGB exteroception at deployment, (ii) distills large video world models into a fast recurrent model for onboard control, (iii) targets hundred-meter-scale urban goal navigation—not racing—and (iv) wraps learning with an auditable multi-signal safety shield.** Table~\ref{tab:related} situates Aerial WAM v2 against the closest lines; AirDreamer~\cite{airdremer2026} is the nearest world-model aerial baseline but assumes depth observations and a different training and verification stack.
