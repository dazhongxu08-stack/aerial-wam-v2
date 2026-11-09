#!/usr/bin/env python3
"""Quality gate for the open-side 0.6 m labels. Exit 0 only when training may start.

A kept probe is one picture where a 0.6 m yaw toward the open side contacts and
an in-place yaw does not. Forward score may be below 0.95. The earlier arc set
is not part of this gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/yao/aerial-wam-v2")
OPEN = ROOT / "artifacts/side_corner_openc13_20261102"
HITS = ROOT / "artifacts/side_corner_branches_20261030"
CLEARS = ROOT / "artifacts/side_corner_clear_20261102"
DEPTH = ROOT / "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
NEED = 24


def thirds(depth56: np.ndarray) -> tuple[float, float, float, float]:
    d = np.asarray(depth56, np.float32)
    hh, ww = d.shape
    rows = d[hh // 4: hh - hh // 4]

    def near(block: np.ndarray) -> float:
        vals = block[np.isfinite(block) & (block > 0.05)]
        return float(vals.min()) if vals.size else 100.0

    left = near(rows[:, : ww // 3])
    right = near(rows[:, ww - ww // 3:])
    c0, c1 = ww // 3, ww - ww // 3
    span = c1 - c0
    inner = near(rows[:, c0 + span // 4: c1 - span // 4])
    center = near(rows[:, c0:c1])
    return left, center, right, inner


def open_sign(depth56: np.ndarray) -> float | None:
    left, center, right, inner = thirds(depth56)
    if center > 4.0 or inner < 8.0:
        return None
    if left <= 4.0 and right >= 8.0:
        return 1.0
    if right <= 4.0 and left >= 8.0:
        return -1.0
    return None


def keep_count(path: Path, rule: str) -> int:
    review = json.loads((path / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    return sum(
        1 for rec in review["probes"]
        if rec.get("train_keep") is not False and rec.get("side_rule") == rule
    )


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import torch

    from experiments.aerial.rl.collect_consequence_dataset import pool_min
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation
    from experiments.aerial.rl.precompute_da3_consequence import decode_frames

    review_path = OPEN / "AUTO_REVIEW.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    log = Path("/tmp/side_openc13.log").read_text(encoding="utf-8", errors="replace")
    reasons: list[str] = []
    if "DONE states=" not in log:
        reasons.append("collector log has no DONE line")
    if "Traceback" in log.split("DONE states=")[-1]:
        reasons.append("traceback after DONE")
    hits_n = keep_count(HITS, "da3_still_high")
    clears_n = keep_count(CLEARS, "da3_still_clear")
    if hits_n < NEED:
        reasons.append(f"old hit set has {hits_n} keeps")
    if clears_n < NEED:
        reasons.append(f"old clear set has {clears_n} keeps")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred = DepthMinPredictor.from_checkpoint(DEPTH, device=dev)
    dropped: list[dict] = []
    for rec in review["probes"]:
        if rec.get("train_keep") is False:
            continue
        why = []
        rid = str(rec.get("route_id"))
        if rec.get("side_rule") != "da3_open_hit":
            why.append(f"side_rule {rec.get('side_rule')}")
        if not rid.startswith("od_train") or rid in VAL:
            why.append(f"route {rid}")
        if float(rec.get("max_replace_err_m") or 0.0) > 1.0:
            why.append(f"replace_err {rec.get('max_replace_err_m')}")
        sign = float(rec.get("open_sign") or 0.0)
        fwd = float(rec.get("fwd_surv") or 0.0)
        if sign == 0.0:
            why.append("open_sign missing")
        if not (0.85 <= fwd <= 0.97):
            why.append(f"fwd_surv {rec.get('fwd_surv')}")
        path = OPEN / "probes" / rec["file"]
        if not path.is_file():
            why.append("missing npz")
            rec["train_keep"] = False
            rec["quality_drop"] = why
            dropped.append({"file": rec.get("file"), "why": why})
            continue
        z = np.load(path)
        cands = np.asarray(z["candidates"], np.float64)
        coll = np.asarray(z["coll_step"])
        fams = list(rec.get("families") or [])
        if cands.ndim != 3 or cands.shape[0] != len(fams) or cands.shape[1] < 10 or cands.shape[2] != 4:
            why.append(f"cand shape {getattr(cands, 'shape', None)} fams {len(fams)}")
        else:
            open_hit = yaw_clear = 0
            for i, name in enumerate(fams):
                rows = np.unique(np.round(cands[i], 5), axis=0)
                a = cands[i, 0]
                held = len(rows) == 1 and abs(a[1]) <= 1e-6 and abs(a[2]) <= 1e-6
                if name == "open_hold":
                    if not held or abs(a[0] - 0.6) > 1e-3 or not (0.04 - 1e-6 <= abs(a[3]) <= 0.09 + 1e-6) or a[3] * sign <= 0:
                        why.append(f"open command {i} {a.tolist()}")
                        break
                    if coll[i] >= 0:
                        open_hit += 1
                    else:
                        why.append(f"open arc {i} did not contact")
                        break
                elif name == "yaw_only":
                    if not held or a[0] >= 0.6:
                        why.append(f"yaw command {i} {a.tolist()}")
                        break
                    if coll[i] < 0:
                        yaw_clear += 1
                    else:
                        why.append(f"yaw arc {i} contacted")
                        break
                else:
                    why.append(f"family {name}")
                    break
            if not why and (open_hit < 1 or yaw_clear < 1):
                why.append(f"pair open_hit {open_hit} yaw_clear {yaw_clear}")
        if why:
            rec["train_keep"] = False
            rec["quality_drop"] = why
            dropped.append({"file": rec["file"], "route_id": rid, "why": why})
            continue
        frames = decode_frames({"jpeg": z["hist_jpeg"], "jpeg_offsets": z["hist_jpeg_offsets"]})
        pred.reset()
        last = None
        for rgb in frames:
            seen = pred.predict_depth(Observation(
                rgb=np.ascontiguousarray(rgb), state=np.zeros(7, np.float32), collided=False))
            if seen is not None:
                last = pool_min(np.asarray(seen, np.float32))
        replay = open_sign(last) if last is not None else None
        if replay is None or replay * sign <= 0:
            why.append("replay picture is not the same open side")
            rec["train_keep"] = False
            rec["quality_drop"] = why
            dropped.append({"file": rec["file"], "route_id": rid, "why": why})

    kept = [p for p in review["probes"] if p.get("train_keep")]
    report = {
        "pass": len(kept) >= NEED and not reasons,
        "pairs": len(kept),
        "need": NEED,
        "dropped": dropped,
        "blocked": reasons,
        "old_hits": hits_n,
        "old_clears": clears_n,
    }
    review_path.write_text(json.dumps(review, indent=1), encoding="utf-8")
    (OPEN / "QUALITY.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
