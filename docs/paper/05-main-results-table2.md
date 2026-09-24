# Table 2 draft — main results (Phase-2)

**Protocol to cite**: `artifacts/seen_airsim16_long_routes.json`, E2 ckpt + `tti_coeff=2.5`, cruise 10 m/s, toward_g (tag-era stack).  
**Acceptance artifact**: `wam_phase2_e2_tti25_full16_20260908.json` · milestone **`phase2-pass-20260908`**.

| System | SR | SPL | IR | SCR | Notes |
|--------|----|-----|----|-----|-------|
| **Ours (Phase-2 PASS)** | **86.7% (13/15 scored)** | — | — | **6.7%** | Tag message: route_idx=8 excluded as spawn anomaly; scored denominator = 15 |
| Naive 13/16 | 81.25% | — | — | — | Includes spawn_fail in denominator — **not** the gate / PASS number |
| Step-G closed-loop (prior) | 93.33% (14/15) | — | 0.80% | 0% | Different protocol — do not mix into Phase-2 row |

### Ablation contrast (same Phase-2 16 routes; V11 polyline stack 2026-09-23)

| Variant | SR | IR | vs mainline |
|---------|----|----|-------------|
| E2a no planner | 0% | 58.1% | planner load-bearing |
| E2b no shield | 62.5% | 0% | shield ≠ sole arrival |
| E2c direct_g | 50% | 75.2% | polyline subgoals matter |

### Action items before camera-ready

- [ ] Attach SPL / IR / Prog from `wam_phase2_e2_tti25_full16_20260908.json` to Ours row
- [ ] Note ablation stack (V11 polyline) vs PASS-era toward_g if protocols differ in methods
- [ ] E3 GT-depth row when eval finishes
