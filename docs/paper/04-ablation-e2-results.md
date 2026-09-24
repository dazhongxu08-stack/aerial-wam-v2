# Table 3 draft — Phase-2 ablations (E2)

**Protocol**: `artifacts/seen_airsim16_long_routes.json`, V11 stack ckpts, cruise 10 m/s, max-steps 600.  
**Host**: ablation-4090, 2026-09-23.

| Variant | Change vs mainline | SR | SPL | IR | SCR | Prog |
|---------|--------------------|----|-----|----|-----|------|
| E1 (cite) | Full stack (planner + shield + polyline) | ~86.7%* | — | — | — | — |
| **E2a** | `--no-planner` (actor-only) | **0/16 (0%)** | 0.0% | **58.1%** | 18.8% | 44.4% |
| **E2b** | `--no-shield` | **10/16 (62.5%)** | 58.8% | **0%** | 12.5% | 80.3% |
| **E2c** | `--subgoal-source direct_g` (no polyline / rolling-global) | **8/16 (50%)** | 49.2% | **75.2%** | 12.5% | 66.2% |

\*E1 number from prior signoff — verify against DECLARE JSON before camera-ready.

### Interpretation

1. Removing ImaginationPlanner collapses arrival → planning is load-bearing for Phase-2 long routes.
2. Removing the ThreeZone shield keeps substantial SR with zero logged intervention → shield audits/caps rather than being the only path to goal.
3. Single-scale `direct_g` hurts vs polyline mainline (50% vs ~87% SR) and drives high shield IR → multi-scale subgoals are material.

### Sources

- `artifacts/e2_ablation_phase2_20260923_091203/e2a_no_planner/full16.json`
- `artifacts/e2_ablation_phase2_20260923_091203/e2b_no_shield/full16.json`
- `artifacts/e2c_direct_g_phase2_20260923_165130/full16.json` — Verdict=FAIL, SR=50%, IR=75.2% (HB=29.0% / GC=46.2%)
