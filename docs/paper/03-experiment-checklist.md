# Experiment checklist (submission readiness)

| ID | Experiment | Status | Notes |
|----|------------|--------|-------|
| E1 | Phase-2 16-route urban benchmark (SR, progress, collision) | ✅ Done | **Cite 86.7% (13/15 scored)** — tag `phase2-pass-20260908`, eval `wam_phase2_e2_tti25_full16_20260908.json`; route_idx=8 spawn anomaly excluded from scored (protocol). Naive 13/16=81.25% is **not** the gate number. |
| E2a | Ablate ImaginationPlanner (actor-only) | ✅ Done | Phase-2 `seen_airsim16_long_routes`: **SR 0/16 (0%)**, IR **58.1%**, SCR 18.8%, Prog 44.4% — `artifacts/e2_ablation_phase2_20260923_091203/e2a_no_planner/full16.json` |
| E2b | Ablate ThreeZone shield (shield-off) | ✅ Done | Same set: **SR 10/16 (62.5%)**, SPL 58.8%, IR **0%**, SCR 12.5%, Prog 80.3% — `.../e2b_no_shield/full16.json` |
| E2c | Ablate subgoals (single-scale `direct_g`) | ✅ Done | **SR 8/16 (50%)**, SPL 49.2%, IR **75.2%**, SCR 12.5%, Prog 66.2% — `artifacts/e2c_direct_g_phase2_20260923_165130/full16.json` |
| E3 | GT depth upper bound vs predicted $\hat{D}$ | ✅ Done | V11 pair: **GT SR 68.8%** vs **pred SR 25.0%** (Δ+43.8 pp); IR 12.4% vs 78.3%. Artifacts `e3_gt_depth_20260925_012031` / `e3_pred_depth_20260925_033522`. See [`06-e3-e4-results.md`](06-e3-e4-results.md). |
| E4 | Horizon $H \in \{1,3,5,10\}$ | ✅ Done | All H full16. **H1 SR 37.5%** (remeshed 2026-09-25); H3 25%; H5 18.8%; H10 31.2%. Root `artifacts/e4_horizon_sweep_phase2_20260924_010441/`. See [`06-e3-e4-results.md`](06-e3-e4-results.md). |
| E5 | Orin real flight ≥10 episodes | 🔲 TODO | Short demo route; log traj + video |
| E6 | Ground sim zero-shot transfer | 🔲 Planned | G0/G1 from work overview |

## E2 takeaway (for Table 3 / discussion)

- **Planner necessary for arrival**: E2a (no ImaginationPlanner) → SR 0% on the same 16 routes.
- **Shield is safety/intervention, not the sole arrival driver**: E2b (shield off) still reaches **62.5%** SR with **IR=0**.
- **Polyline / multi-scale subgoals help**: E2c (`direct_g`) drops to **50%** SR vs E1 ~87%, with high IR **75.2%** (shield working hard).
- Artifacts: `e2_ablation_phase2_20260923_091203/` (E2a/b), `e2c_direct_g_phase2_20260923_165130/` (E2c) on ablation-4090.

## E3 / E4 takeaway (draft)

- **E3 depth pair (V11)**: GT **68.8%** vs predicted **25.0%** (IR 12.4% → 78.3%) — monocular $\hat{D}$ is a major bottleneck on this protocol.
- **E4 horizon**: SR order **H1 37.5% > H10 31% > H3 25% > H5 19%**; larger $H$ is not monotonically better.

## E5 protocol sketch (copy to methods)

- Platform: Orin + Pixhawk 6C + USB camera (1280×720)
- Task: short goal approach 15–25 m (`RUNBOOK_orin_vgoal_demo`)
- Episodes: ≥10, props on, manual kill switch armed
- Metrics: arrival (≤5 m), collision, takeover count, latency (RGB→cmd)
- Artifacts: `artifacts/orin_deploy/run_*`, MJPEG log optional

## Tables still to generate

- [ ] Table 1: Related work (LaTeX in `01-related-work.md`)
- [x] Table 2: Main results vs baselines — draft in `05-main-results-table2.md` (E3 row filled)
- [x] Table 3: Ablations (E2) — E2a/E2b/E2c numbers locked
- [x] E3 / E4 detail — [`06-e3-e4-results.md`](06-e3-e4-results.md)
- [ ] Figure 1: System overview
- [ ] Figure 2: Distillation pipeline
- [ ] Figure 3: Qualitative trajectories (16-route overlay)
