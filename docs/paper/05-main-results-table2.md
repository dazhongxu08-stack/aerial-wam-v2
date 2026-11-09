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

### Depth / horizon (V11 stack; see [`06-e3-e4-results.md`](06-e3-e4-results.md))

| Variant | SR | SPL | IR | Notes |
|---------|----|-----|----|-------|
| **E3 GT depth** | **68.8% (11/16)** | **68.0%** | **12.4%** | `--use-gt-depth`; not a deploy claim |
| **E3 predicted $\hat{D}$** | **25.0% (4/16)** | **24.5%** | **78.3%** | fair V11 pair to GT |
| E4 H=1 | **37.5% (6/16)** | 35.4% | 63.1% | horizon sweep (same stack class; separate run) |
| E4 H=3 | 25.0% | 24.5% | 72.4% | full16 |
| E4 H=5 | 18.8% | 18.5% | 76.0% | full16 |
| E4 H=10 | 31.2% | 30.4% | 68.5% | full16 |

### Action items before camera-ready

- [ ] Attach SPL / IR / Prog from `wam_phase2_e2_tti25_full16_20260908.json` to Ours row
- [ ] Note ablation stack (V11 polyline) vs PASS-era toward_g if protocols differ in methods
- [x] E3 GT-depth row locked (`e3_gt_depth_20260925_012031`)
- [x] V11 predicted-$\hat{D}$ full16 as fair pair to E3 (`e3_pred_depth_20260925_033522`, SR 25%)
- [x] Re-run E4 H=1 chunk_1 (routes 4–7) — full16 SR 37.5%
