#!/usr/bin/env python3
"""Keep/drop interior start/goal candidates in AirSim with the eval spawn path.

A camera inside a building mesh sees out through back faces, so depth alone
cannot tell a buried spawn. Each end is spawned with the same
``reset_with_spawn_retries`` call as ``wam_phase2_long_eval`` (lift to
``min_spawn_z``, +``spawn_z_retry_m`` per retry, forward clearance), and the goal
is spawned the same way on the reversed segment at the height the start used.

A route whose straight segment is clear at spawn height needs no avoidance, so
the GT depth toward the goal is recorded. ``blocked`` means the central box
facing the goal is nearer than the chord.

Writes ``AUTO_REVIEW.json`` (every candidate with keep/drop reasons), a contact
sheet of start views, and the kept train / held-out annotations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

MIN_SPAWN_INLAND_M = 100.0
MIN_PATH_INLAND_M = 80.0


def _center_box(depth: np.ndarray, frac: float) -> np.ndarray:
    h, w = depth.shape[:2]
    dh, dw = max(1, int(h * frac)), max(1, int(w * frac))
    r0, c0 = (h - dh) // 2, (w - dw) // 2
    return depth[r0 : r0 + dh, c0 : c0 + dw]


def goal_line_depth(depth: np.ndarray) -> Tuple[float, float]:
    """(median, 5th pct) GT depth in the central 10% box facing the goal."""
    box = _center_box(np.asarray(depth, dtype=np.float64), 0.10)
    v = box[np.isfinite(box) & (box > 0.05)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(np.median(v)), float(np.percentile(v, 5))


def geography_reasons(meta: Dict[str, Any]) -> List[str]:
    out = []
    if not meta.get("in_urban"):
        out.append("not_urban")
    if not meta.get("in_interior"):
        out.append("not_interior")
    if meta.get("in_water"):
        out.append("water")
    if float(meta.get("spawn_inland_m", 0.0)) < MIN_SPAWN_INLAND_M:
        out.append("spawn_inland<100")
    if float(meta.get("goal_inland_m", 0.0)) < MIN_SPAWN_INLAND_M:
        out.append("goal_inland<100")
    if float(meta.get("path_min_inland_m", 0.0)) < MIN_PATH_INLAND_M:
        out.append("path_inland<80")
    return out


def urban_near_frac(env: Any, x: float, y: float, z_up: float) -> float:
    """Mean above-horizon near-depth fraction over four headings (see scan_urban_mask_airsim)."""
    import airsim

    from experiments.aerial.scripts.scan_urban_mask_airsim import above_horizon_near_frac, grab_depth

    client = getattr(env, "client", None) or getattr(env, "_client", None)
    if client is None:
        client = airsim.MultirotorClient()
        client.confirmConnection()
    fr = [
        above_horizon_near_frac(grab_depth(client, airsim, x, y, z_up, yaw, 0.12), 150.0)
        for yaw in (0.0, 90.0, 180.0, 270.0)
    ]
    return float(np.mean(fr))


def _spawn(env: Any, pos: List[List[float]], yaw: float, args: argparse.Namespace):
    from experiments.aerial.rl.spawn_utils import reset_with_spawn_retries

    ep = {"pos": pos, "yaw": [float(yaw)] * len(pos), "gpt_instruction": ""}
    return reset_with_spawn_retries(
        env,
        ep,
        min_spawn_z=float(args.min_spawn_z),
        spawn_z_retry_m=float(args.spawn_z_retry_m),
        spawn_z_max_retries=int(args.spawn_z_max_retries),
        spawn_tol_m=float(args.spawn_tol_m),
        mock=False,
        min_spawn_clear_m=float(args.min_spawn_clear_m),
    )


def review_one(env: Any, ep: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any], Any]:
    meta = ep["complexity_meta"]
    pos = [list(map(float, p)) for p in ep["pos"]]
    yaw = float(ep["yaw"][0])
    chord = float(meta["chord_m"])
    reasons = geography_reasons(meta)
    row: Dict[str, Any] = {
        "route_id": ep["route_id"],
        "split": meta["split"],
        "region": meta["region"],
        "chord_m": chord,
        "heading_deg": meta["heading_deg"],
    }
    used, obs, err, failed = _spawn(env, pos, yaw, args)
    used_start = np.asarray(used["pos"], dtype=np.float64)[0]
    spawn_z = float(used_start[2])
    start_shift = float(np.linalg.norm(used_start[:2] - np.asarray(pos[0][:2])))
    row.update({
        "spawn_failed": bool(failed),
        "spawn_err_m": round(float(err), 2),
        "spawn_z": spawn_z,
        "start_xy_shift_m": round(start_shift, 1),
    })
    rgb = None
    if failed:
        reasons.append("spawn_fail")
    elif start_shift > 1.0:
        reasons.append("start_xy_nudged")
    else:
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        med, p5 = goal_line_depth(obs.depth)
        row["goal_line_depth_med_m"] = round(med, 1) if np.isfinite(med) else None
        row["goal_line_depth_p5_m"] = round(p5, 1) if np.isfinite(p5) else None
        row["blocked"] = bool(np.isfinite(p5) and p5 < chord - float(args.block_margin_m))
        rev = [[p[0], p[1], spawn_z] for p in reversed(pos)]
        gused, _, gerr, gfailed = _spawn(
            env, rev, yaw + math.pi, argparse.Namespace(**{**vars(args), "min_spawn_z": spawn_z, "spawn_z_max_retries": 0}),
        )
        gstart = np.asarray(gused["pos"], dtype=np.float64)[0]
        goal_shift = float(np.linalg.norm(gstart[:2] - np.asarray(rev[0][:2])))
        goal_dz = float(gstart[2] - spawn_z)
        row.update({
            "goal_spawn_failed": bool(gfailed),
            "goal_spawn_err_m": round(float(gerr), 2),
            "goal_xy_shift_m": round(goal_shift, 1),
            "goal_dz_m": round(goal_dz, 1),
        })
        if gfailed or goal_shift > 1.0 or abs(goal_dz) > 0.5:
            reasons.append("goal_in_geometry")
        s_frac = urban_near_frac(env, pos[0][0], pos[0][1], spawn_z)
        g_frac = urban_near_frac(env, pos[-1][0], pos[-1][1], spawn_z)
        row.update({"start_urban_frac": round(s_frac, 3), "goal_urban_frac": round(g_frac, 3)})
        if s_frac < float(args.min_urban_frac):
            reasons.append("start_not_city_in_render")
        if g_frac < float(args.min_urban_frac):
            reasons.append("goal_not_city_in_render")
    row["keep"] = not reasons
    row["drop_reasons"] = reasons
    return row, rgb


def contact_sheet(tiles: List[Tuple[str, np.ndarray]], out: Path, cols: int = 6) -> None:
    import cv2

    if not tiles:
        return
    tw, th = 224, 224
    rows = int(math.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * (th + 18), cols * tw, 3), dtype=np.uint8)
    for i, (label, rgb) in enumerate(tiles):
        r, c = divmod(i, cols)
        img = cv2.resize(rgb, (tw, th))[:, :, ::-1]
        y0 = r * (th + 18)
        sheet[y0 + 18 : y0 + 18 + th, c * tw : (c + 1) * tw] = img
        cv2.putText(sheet, label, (c * tw + 4, y0 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(out), sheet)


def write_annotation(eps: List[Dict[str, Any]], out: Path, split: str) -> None:
    for i, ep in enumerate(eps):
        ep["route_idx"] = i
    payload = {
        "protocol_version": f"interior_od_{split}_v1",
        "scene": "outdoor_complex",
        "n_episodes": len(eps),
        "template": "interior_od_v1",
        "episodes": eps,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--candidates",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_od_candidates.json",
    )
    ap.add_argument("--out-dir", default="experiments/aerial/rl/artifacts/interior_od_routes_20260930")
    ap.add_argument(
        "--ann-prefix",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_od",
    )
    ap.add_argument("--config", default="configs/_tmp_feasible_side_h125.yaml")
    ap.add_argument("--min-spawn-z", type=float, default=24.0)
    ap.add_argument("--spawn-z-retry-m", type=float, default=5.0)
    ap.add_argument("--spawn-z-max-retries", type=int, default=4)
    ap.add_argument("--spawn-tol-m", type=float, default=12.0)
    ap.add_argument("--min-spawn-clear-m", type=float, default=5.0)
    ap.add_argument("--block-margin-m", type=float, default=5.0)
    ap.add_argument("--min-urban-frac", type=float, default=0.15)
    ap.add_argument("--min-train-blocked", type=int, default=16)
    ap.add_argument("--min-holdout-blocked", type=int, default=8)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import yaml

    from experiments.aerial.rl.train_rl import _build_env

    cand = json.loads((root / args.candidates).read_text(encoding="utf-8"))
    cfg = yaml.safe_load((root / args.config).read_text()) or {}
    env_cfg = dict(cfg.get("env") or {})
    env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
    env = _build_env(env_cfg)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    kept: Dict[str, List[Dict[str, Any]]] = {"train": [], "holdout": []}
    for split in ("train", "holdout"):
        tiles: List[Tuple[str, np.ndarray]] = []
        for ep in cand[split]:
            row, rgb = review_one(env, ep, args)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if rgb is not None:
                tag = "B" if row.get("blocked") else "o"
                tiles.append((f"{row['route_id']} {tag} {row['chord_m']:.0f}m", rgb))
            if row["keep"]:
                e = dict(ep)
                e["complexity_meta"] = {
                    **ep["complexity_meta"],
                    "spawn_z_ok": row["spawn_z"],
                    "blocked": row["blocked"],
                    "goal_line_depth_p5_m": row["goal_line_depth_p5_m"],
                }
                kept[split].append(e)
        contact_sheet(tiles, out_dir / f"start_views_{split}.jpg")

    summary: Dict[str, Any] = {}
    for split in ("train", "holdout"):
        ks = kept[split]
        blocked = [e for e in ks if e["complexity_meta"]["blocked"]]
        clear = [e for e in ks if not e["complexity_meta"]["blocked"]]
        clear = clear[: len(blocked) // 2]
        final = blocked + clear
        need = int(args.min_train_blocked if split == "train" else args.min_holdout_blocked)
        summary[split] = {
            "candidates": len(cand[split]),
            "kept": len(ks),
            "blocked": len(blocked),
            "clear_in_annotation": len(clear),
            "annotation_routes": len(final),
            "need_blocked": need,
            "gate": len(blocked) >= need,
        }
        write_annotation(final, root / f"{args.ann_prefix}_{split}.json", split)
    gate = all(v["gate"] for v in summary.values())
    review = {"gate_pass": gate, "summary": summary, "rows": rows, "args": vars(args)}
    (out_dir / "AUTO_REVIEW.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), "GATE", gate, flush=True)
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
