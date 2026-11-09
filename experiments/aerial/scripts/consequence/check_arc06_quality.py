#!/usr/bin/env python3
"""Quality gate for the held 0.6 m arc collection. Exit 0 only when training may start.

Drops probes that are not the side picture, or whose replayed forward score is
already below 0.95, by setting train_keep false. Training starts only when at
least 24 contacts and 24 non-contacts remain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/yao/aerial-wam-v2")
ARC = ROOT / "artifacts/side_corner_arc06_20261102"
HITS = ROOT / "artifacts/side_corner_branches_20261030"
CLEARS = ROOT / "artifacts/side_corner_clear_20261102"
ART = ROOT / "experiments/aerial/rl/artifacts"
CKPTS = [ART / f"shorew03_krs6_20261013/c_s{s}/consequence_best.pt" for s in (218, 219, 220)]
DEPTH = ART / "depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
NEED = 24


def side_corner(depth56: np.ndarray) -> bool:
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
    one = (left <= 4.0 and right >= 8.0) or (right <= 4.0 and left >= 8.0)
    return bool(one and inner >= 8.0)


def keep_count(path: Path, rule: str | None = None) -> int:
    review = json.loads((path / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    n = 0
    for rec in review["probes"]:
        if rec.get("train_keep") is False:
            continue
        if rule is not None and rec.get("side_rule") != rule:
            continue
        n += 1
    return n


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import torch

    from experiments.aerial.rl.collect_consequence_dataset import pool_min
    from experiments.aerial.rl.consequence_model import depth_input, goal_features, load_checkpoint, survival
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation
    from experiments.aerial.rl.precompute_da3_consequence import decode_frames

    review_path = ARC / "AUTO_REVIEW.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    log = Path("/tmp/side_arc06.log").read_text(encoding="utf-8", errors="replace")
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
    models = [load_checkpoint(p, device=dev) for p in CKPTS]
    for m in models:
        m.eval()
    pred = DepthMinPredictor.from_checkpoint(DEPTH, device=dev)
    H = models[0].horizon

    def fwd_surv(depth56: np.ndarray, state: np.ndarray, goal: np.ndarray, prev: np.ndarray) -> float:
        action = np.zeros((H, 4), np.float32)
        action[:, 0] = 0.8
        d = torch.as_tensor(depth_input(depth56)[None], device=dev)
        g = torch.as_tensor(goal_features(state, goal)[None], device=dev)
        pa = torch.as_tensor(prev[None], device=dev)
        a = torch.as_tensor(action[None], device=dev)
        scores = []
        with torch.no_grad():
            for m in models:
                out = m(d, g, pa, a)
                scores.append(float(survival(out["hazard_logit"])[..., -1].detach().cpu()))
        return float(np.mean(scores))

    dropped: list[dict] = []
    for rec in review["probes"]:
        if rec.get("train_keep") is False:
            continue
        why = []
        if rec.get("side_rule") != "da3_arc06":
            why.append(f"side_rule {rec.get('side_rule')}")
        if float(rec.get("fwd_surv") or 0.0) < 0.95:
            why.append(f"stored fwd_surv {rec.get('fwd_surv')}")
        rid = str(rec.get("route_id"))
        if not rid.startswith("od_train") or rid in VAL:
            why.append(f"route {rid}")
        if float(rec.get("max_replace_err_m") or 0.0) > 1.0:
            why.append(f"replace_err {rec.get('max_replace_err_m')}")
        path = ARC / "probes" / rec["file"]
        if not path.is_file():
            why.append("missing npz")
            rec["train_keep"] = False
            rec["quality_drop"] = why
            dropped.append({"file": rec.get("file"), "why": why})
            continue
        z = np.load(path)
        cands = np.asarray(z["candidates"], np.float64)
        coll = np.asarray(z["coll_step"])
        if cands.ndim != 3 or cands.shape[1] < 10 or cands.shape[2] != 4:
            why.append(f"cand shape {cands.shape}")
        else:
            for i in range(len(cands)):
                rows = np.unique(np.round(cands[i], 5), axis=0)
                a = cands[i, 0]
                if len(rows) != 1 or abs(a[0] - 0.6) > 1e-3 or abs(a[1]) > 1e-6 or abs(a[2]) > 1e-6 or abs(a[3]) > 0.10 + 1e-6:
                    why.append(f"command {i} {a.tolist()}")
                    break
        hit = bool(np.any(coll >= 0))
        if bool(rec.get("arc_hit")) != hit:
            why.append("arc_hit label mismatch")
        frames = decode_frames({"jpeg": z["hist_jpeg"], "jpeg_offsets": z["hist_jpeg_offsets"]})
        pred.reset()
        last = None
        for rgb in frames:
            seen = pred.predict_depth(Observation(
                rgb=np.ascontiguousarray(rgb), state=np.zeros(7, np.float32), collided=False))
            if seen is not None:
                last = pool_min(np.asarray(seen, np.float32))
        if last is None or not side_corner(last):
            why.append("replay picture is not the side corner")
        else:
            hist = np.asarray(z["hist_action"], np.float32)
            prev = hist[-2] if len(hist) >= 2 else np.zeros(4, np.float32)
            surv = fwd_surv(last, np.asarray(z["state"], np.float64), np.asarray(z["goal"], np.float64), prev)
            rec["replay_fwd_surv"] = round(surv, 4)
            if surv < 0.95:
                why.append(f"replay fwd_surv {surv:.3f}")
        if why:
            rec["train_keep"] = False
            rec["quality_drop"] = why
            dropped.append({"file": rec["file"], "route_id": rid, "why": why})

    clears = [p for p in review["probes"] if p.get("train_keep") and not p.get("arc_hit")]
    for rec in clears[NEED:]:
        rec["train_keep"] = False
        rec["quality_drop"] = ["extra clear held out so the hit and clear counts stay matched"]
        dropped.append({"file": rec["file"], "route_id": rec.get("route_id"), "why": rec["quality_drop"]})
    kept = [p for p in review["probes"] if p.get("train_keep")]
    hit_n = sum(1 for p in kept if p.get("arc_hit"))
    clear_n = sum(1 for p in kept if not p.get("arc_hit"))
    report = {
        "pass": hit_n >= NEED and clear_n >= NEED and not reasons,
        "hit": hit_n,
        "clear": clear_n,
        "need": NEED,
        "dropped": dropped,
        "blocked": reasons,
        "old_hits": hits_n,
        "old_clears": clears_n,
    }
    review_path.write_text(json.dumps(review, indent=1), encoding="utf-8")
    (ARC / "QUALITY.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    if report["pass"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
