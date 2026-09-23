# Experiment checklist (submission readiness)

| ID | Experiment | Status | Notes |
|----|------------|--------|-------|
| E1 | Phase-2 16-route urban benchmark (SR, progress, collision) | ✅ Done | 86.7% cited — verify against signoff JSON |
| E2a | Ablate ImaginationPlanner (actor-only) | 🔄 Running | **Phase-2 only**: `artifacts/seen_airsim16_long_routes.json` (not urban-complex / inland hard) |
| E2b | Ablate ThreeZone shield (shield-off) | 🔄 Queued | Same Phase-2 16-route set as E2a / E1 |
| E2c | Ablate subgoals (single-scale goals) | 🔲 TODO | Long-route regression |
| E3 | GT depth upper bound vs predicted $\hat{D}$ | 🔲 TODO | Quantify monocular contract cost |
| E4 | Horizon $H \in \{1,3,5,10\}$ | 🔲 Partial | Planner sweep |
| E5 | Orin real flight ≥10 episodes | 🔲 **Blocker** | Short demo route; log traj + video |
| E6 | Ground sim zero-shot transfer | 🔲 Planned | G0/G1 from work overview |

## E5 protocol sketch (copy to methods)

- Platform: Orin + Pixhawk 6C + USB camera (1280×720)
- Task: short goal approach 15–25 m (`RUNBOOK_orin_vgoal_demo`)
- Episodes: ≥10, props on, manual kill switch armed
- Metrics: arrival (≤5 m), collision, takeover count, latency (RGB→cmd)
- Artifacts: `artifacts/orin_deploy/run_*`, MJPEG log optional

## Tables still to generate

- [ ] Table 1: Related work (LaTeX in `01-related-work.md`)
- [ ] Table 2: Main results vs baselines
- [ ] Table 3: Ablations (E2)
- [ ] Figure 1: System overview
- [ ] Figure 2: Distillation pipeline
- [ ] Figure 3: Qualitative trajectories (16-route overlay)
