# Abstract & Contributions (draft)

---

## Title options

1. **Deployable Pure-Vision Aerial Navigation with a Distilled Fast World Model, Imagination Planning, and Auditable Safety Shields** *(recommended)*

2. **Aerial WAM: World-Model Navigation for Long-Range Monocular UAVs via Offline Video Distillation and Layered Imagination**

3. **From Video World Models to Onboard Control: Pure-Vision Goal Navigation for Urban Aerial Agents**

---

## Abstract — conference version (~180 words)

```text
Long-range autonomous flight in unknown urban environments remains difficult under
the sensing and compute budgets of small multirotors: goals are partially observed,
obstacle layout must be inferred online, and global SLAM or depth hardware are often
unavailable or undesirable. We present Aerial WAM v2, a deployable vision-only
navigation stack that couples (i) offline distillation from a large video–action
world model into a fast Dreamer-style recurrent state-space model (RSSM), (ii)
hierarchical decision-making with adaptive subgoals and short-horizon imagination
planning in latent space, and (iii) a three-zone safety shield driven by predicted
depth, time-to-contact, and collision risk. At deployment, exteroception is
restricted to a single monocular RGB camera; metric depth is predicted, not sensed,
and no loop-closed global map is used. In photorealistic urban simulation, our
system passes formal Phase-2 acceptance on a 16-route outdoor benchmark with
86.7% success rate and demonstrates 153.7 m closed-loop traversal on Step-G
routes. We further describe an Orin companion-computer deployment with Pixhawk
offboard control and a unified sim-to-real interface. Real-world outdoor flight
statistics are ongoing work. Code, evaluation protocols, and video evidence are
released with the project repository.
```

---

## Abstract — short (arXiv / 150 words)

```text
We study goal-first aerial navigation under a strict monocular sensing contract:
one RGB camera for exteroception, with IMU and altitude as proprioception only.
Aerial WAM v2 trains a large Wan-class video world model offline and distills it
into a compact RSSM for 5–10 Hz onboard control. A latent actor is guided by
adaptive subgoals (20–55 m) and imagination rollouts (horizon 5), while a
three-zone shield enforces conservative motion using predicted depth, optical-flow
time-to-contact, and collision probability. On a 16-route urban AirSim benchmark,
the system achieves 86.7% arrival rate under a frozen acceptance protocol; Step-G
evaluation reaches 93.3% on 15 routes with routes up to 153.7 m. We report
simulation results, ablation directions, and an Orin–Pixhawk deployment stack;
outdoor real-flight success rates are left to future reporting.
```

---

## Contributions (for Introduction — bullet list)

```latex
\begin{itemize}
  \item \textbf{Sensing contract.} We formulate urban aerial navigation as a POMDP
        with \emph{monocular RGB-only} exteroception at deployment: no depth camera,
        stereo rig, LiDAR, or loop-closed global SLAM map.
  \item \textbf{Train heavy, fly light.} A Wan2.2-scale video--action model provides
        offline representation pretraining; runtime control uses only a distilled
        \textbf{TorchRSSM}, DA3 depth head, and latent actor on edge hardware.
  \item \textbf{Layered imagination stack.} \texttt{AdaptiveSubgoal} decomposes
        50--200\,m missions; \texttt{ImaginationPlanner} ($H{=}5$) scores short
        action sequences in latent space before execution.
  \item \textbf{Auditable safety.} A \textbf{ThreeZoneSafetyShield} fuses predicted
        depth $\hat{D}$, FOE-based time-to-contact $\tau$, and collision head
        $p_{\mathrm{coll}}$ with hysteresis, under a signed gate protocol (V0--V4).
  \item \textbf{Benchmark \& deployment.} Phase-2 acceptance on 16 urban routes
        (86.7\% SR); Step-G routes to 153.7\,m; open Orin deploy stack with
        sim-identical interfaces (real-flight table: \textit{TBD}).
\end{itemize}
```

---

## Keywords

`world models`; `aerial robotics`; `monocular vision`; `model-based reinforcement learning`; `imagination planning`; `safety shields`; `sim-to-real deployment`

---

## LaTeX abstract block (paste-ready)

```latex
\begin{abstract}
Long-range autonomous flight in unknown urban environments remains difficult under
the sensing and compute budgets of small multirotors: goals are partially observed,
obstacle layout must be inferred online, and global SLAM or depth hardware are often
unavailable or undesirable. We present \emph{Aerial WAM v2}, a deployable vision-only
navigation stack that couples (i)~offline distillation from a large video--action
world model into a fast Dreamer-style recurrent state-space model (RSSM),
(ii)~hierarchical decision-making with adaptive subgoals and short-horizon imagination
planning in latent space, and (iii)~a three-zone safety shield driven by predicted
depth, time-to-contact, and collision risk. At deployment, exteroception is
restricted to a single monocular RGB camera; metric depth is predicted, not sensed,
and no loop-closed global map is used. In photorealistic urban simulation, our
system passes formal Phase-2 acceptance on a 16-route outdoor benchmark with
\textbf{86.7\%} success rate and demonstrates \textbf{153.7\,m} closed-loop
traversal on Step-G routes. We further describe an NVIDIA Orin companion-computer
deployment with Pixhawk offboard control and a unified sim-to-real interface.
Real-world outdoor flight statistics are ongoing work.
\end{abstract}
```

---

## Numbers to keep consistent across paper

| Metric | Value | Source |
|--------|-------|--------|
| Phase-2 16-route SR | **86.7%** (13/15 in one report; verify 13/16 = 81.25% vs 86.7% — use handover declare) | `WAM_PHASE2_SIGNOFF` / work overview |
| Step-G 16-route SR | **93.33%** (14/15) | Step G declare 2026-08-28 |
| Step-G mean progress | **97.52%** | same |
| Step-G max route | **153.7 m** | Route 06 |
| Phase-2 long route example | Route 15, 200 s HUD | video V3 |
| Control rate | 5–10 Hz | config |
| Imagination horizon | H=5 (cap 15) | planner |
| Subgoal range | 20–55 m | AdaptiveSubgoal |
| RSSM latent dim | 1536 (h‖z) | architecture table |
| Image / WM input | 224×224 RGB | frozen spec |

> **Note:** Reconcile 86.7% vs 81.25% (13/16) before submission — cite the signed acceptance document only.

---

## One-sentence pitch (for rebuttals)

> We are not claiming the first Dreamer drone; we claim the first **audited, deployable, monocular-only, long-range urban** stack that **distills video world models into an onboard RSSM** with **layered imagination and a signed safety gate protocol**.
