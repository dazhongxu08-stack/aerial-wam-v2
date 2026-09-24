# Table 2 draft — main results (Phase-2)

**Protocol to cite**: `artifacts/seen_airsim16_long_routes.json`, V11 stack (polyline + rolling-global + planner H=1 + ThreeZone), cruise 10 m/s.

| System | SR | SPL | IR | SCR | Notes |
|--------|----|-----|----|-----|-------|
| **Ours (Phase-2 mainline)** | **81.25% (13/16)** | — | — | — | Authoritative: `wam_phase2_e2_tti25_full16_20260908` / work overview |
| Step-G closed-loop (prior) | 93.33% (14/15)* | — | 0.80% | 0% | Different protocol / route set — do not mix into Phase-2 row |
| Legacy marketing “86.7%” | — | — | — | — | **Do not cite** — reconcile artifact; superseded by 13/16 |

\*Step-G DECLARE 2026-08-28; not the Phase-2 16 outdoor-long gate.

### Ablation contrast (same Phase-2 16 routes)

| Variant | SR | IR | vs mainline |
|---------|----|----|-------------|
| E2a no planner | 0% | 58.1% | planner load-bearing |
| E2b no shield | 62.5% | 0% | shield ≠ sole arrival |
| E2c direct_g | 50% | 75.2% | polyline subgoals matter |

### Action items before camera-ready

- [ ] Pull `wam_phase2_e2_tti25_full16_20260908` JSON (or re-run V11 full16) and fill SPL / IR / SCR / Prog for Ours row
- [ ] Add baseline columns (e.g. actor-only already in E2a; open-loop teacher if any)
- [ ] E3 GT-depth row when overnight eval finishes
