# E3 / E4 results — Phase-2 (ablation-4090)

**Protocol**: `artifacts/seen_airsim16_long_routes.json`, V11 polyline stack, cruise 10 m/s, max-steps 600, `tti_coeff=2.5`.  
**Host**: ablation-4090 (`.11` via 125).

---

## E3 — GT depth vs predicted $\hat{D}$ (paired V11)

Same Phase-2 16 routes + V11 stack; only depth source differs.

| Arm | Artifact | SR | SPL | IR | SCR | Prog |
|-----|----------|----|-----|----|-----|------|
| **GT depth** (`--use-gt-depth`) | `e3_gt_depth_20260925_012031` | **68.8% (11/16)** | **68.0%** | **12.4%** | 12.5% | 80.4% |
| **Predicted $\hat{D}$** | `e3_pred_depth_20260925_033522` | **25.0% (4/16)** | **24.5%** | **78.3%** | 6.2% | 56.7% |

**Δ (GT − pred)**: SR **+43.8 pp**, SPL **+43.5 pp**, IR **−65.9 pp**.

### Details

**GT** — `DONE` @ 2026-09-25 03:04 UTC (~1h45m). Failures: routes 3, 7, 9, 10, 15.  
**Pred** — `DONE` @ 2026-09-25 07:47 UTC (followup chain after H1 remesh). Arrivals: routes 0/1-indexed log 01, 03, 09, 14 → 4/16.

### Caveat vs E1 cite

E1 PASS cite is **86.7% (13/15 scored)** on the toward_g / tag-era stack (`phase2-pass-20260908`). Both E3 arms here are **V11 polyline**. Do **not** mix E1 86.7% into the GT-vs-pred delta.

### Note vs E4 H=1

E4 H=1 is also V11 + predicted $\hat{D}$ + H=1 but scored **37.5%** on a separate sweep (incl. remeshed chunk_1). Prefer citing the **dedicated E3 pred arm (25%)** as the fair pair to GT; treat E4 H=1 as the horizon-sweep row (run variance / recover timing may differ).

### Interpretation (draft)

- Matched V11 pair shows **large depth-sensing gap**: GT depth nearly triples SR and collapses IR → predicted $\hat{D}$ is a major bottleneck under this stack.
- Residual GT failures (e.g. route 15 spawn-class) are not all depth-limited.
- Paper framing: E3 = **contract cost of monocular depth** on the ablation protocol.

---

## E4 — Planner horizon $H \in \{1,3,5,10\}$

**Artifact**: `artifacts/e4_horizon_sweep_phase2_20260924_010441/` (`DONE.txt` @ 2026-09-24 13:48 UTC).  
Default mainline uses **H=1** (`--planner-horizon 1`).

| $H$ | n scored | SR | SPL | IR | SCR | Prog | Notes |
|-----|----------|----|-----|----|-----|------|-------|
| **1** | **16** | **37.5% (6/16)** | 35.4% | 63.1% | 12.5% | 73.3% | full16 after 2026-09-25 chunk_1 remesh |
| **3** | 16 | **25.0% (4/16)** | 24.5% | 72.4% | 12.5% | 58.7% | full16 |
| **5** | 16 | **18.8% (3/16)** | 18.5% | 76.0% | 6.3% | 50.7% | full16 |
| **10** | 16 | **31.2% (5/16)** | 30.4% | 68.5% | 12.5% | 57.6% | full16 |

### Interpretation (draft)

- All four horizons now full16. **SR order**: H1 (37.5%) > H10 (31%) > H3 (25%) > H5 (19%) — larger $H$ is not monotonically better; default H=1 is best among the sweep.
- H=10 recovers vs H=5 but still below H=1.
- Discussion hook: open-loop imagination horizon trades off compute vs myopic control; sweet spot at **H=1** on this urban set.

### Follow-ups

- [x] Re-run E4 **H=1** batch routes `4,5,6,7` → merge full16 (2026-09-25; SR 37.5%)
- [x] V11 predicted-$\hat{D}$ full16 — `artifacts/e3_pred_depth_20260925_033522` (**SR 25.0%**, IR 78.3%)
