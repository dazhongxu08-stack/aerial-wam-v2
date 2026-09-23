# Table 3 draft — Phase-2 ablations (E2)

**Protocol**: `artifacts/seen_airsim16_long_routes.json`, V11 stack ckpts, cruise 10 m/s, max-steps 600.  
**Host run**: ablation-4090 (`artifacts/e2_ablation_phase2_20260923_091203/`), 2026-09-23.

| Variant | Change vs mainline | SR | SPL | IR | SCR | Prog |
|---------|--------------------|----|-----|----|-----|------|
| E1 (cite) | Full stack (planner + shield + polyline) | ~86.7%* | — | — | — | — |
| **E2a** | `--no-planner` (actor-only) | **0/16 (0%)** | 0.0% | **58.1%** | 18.8% | 44.4% |
| **E2b** | `--no-shield` | **10/16 (62.5%)** | 58.8% | **0%** | 12.5% | 80.3% |
| E2c | `--subgoal-source direct_g` (no polyline / rolling-global) | 🔄 overnight | | | | |

\*E1 number from prior signoff — verify against DECLARE JSON before camera-ready.

### Interpretation

1. Removing ImaginationPlanner collapses arrival → planning is load-bearing for Phase-2 long routes.
2. Removing the ThreeZone shield keeps substantial SR with zero logged intervention → shield audits/caps rather than being the only path to goal; expect higher collision severity vs shielded E1 (fill SCR delta in final table).
3. E2c tests whether multi-scale polyline subgoals matter vs single-scale `direct_g`.

### Sources

- `e2a_no_planner/full16.json` — Verdict=FAIL, IR=58.1% (HB=44.2% / GC=13.9%)
- `e2b_no_shield/full16.json` — Verdict=FAIL, SR=62.5%, IR=0%
- `PAPER_SUMMARY.json` — per-route arrived flags
