#!/usr/bin/env python3
"""Build Phase-3 outdoor-complex annotation from selected complex routes."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("build_phase3_outdoor_complex")


def main() -> int:
    p = argparse.ArgumentParser(description="Build outdoor-complex Phase-3 annotation")
    p.add_argument("--complex", default="artifacts/seen_airsim16_complex_routes.json")
    p.add_argument(
        "--out",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json",
    )
    p.add_argument("--regenerate", action="store_true", help="Run select_outdoor_complex_routes.py first")
    p.add_argument("--input-m1a20", default="artifacts/seen_airsim16_m1a20.json")
    p.add_argument("--n-select", type=int, default=6)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if args.regenerate:
        cmd = [
            sys.executable,
            "-m",
            "experiments.aerial.scripts.select_outdoor_complex_routes",
            "--input",
            str(args.input_m1a20),
            "--out",
            str(args.complex),
            "--n-select",
            str(args.n_select),
        ]
        subprocess.check_call(cmd, cwd=str(root))

    from experiments.aerial.phase3_unified.mixed_corpus import tag_outdoor_routes, _load_routes

    complex_path = root / args.complex
    routes = _load_routes(complex_path)
    tagged = tag_outdoor_routes(routes)
    min_fly_spawn_z = 12.0
    for ep in tagged:
        ep["scene"] = "outdoor_complex"
        if "complexity" in ep:
            ep["complexity_meta"] = ep.pop("complexity")
        pos = ep.get("pos") or []
        if pos and float(pos[0][2]) < 2.0:
            dz = min_fly_spawn_z - float(pos[0][2])
            for p in pos:
                p[2] = float(p[2]) + dz
            meta = ep.setdefault("complexity_meta", {})
            meta["spawn_z_lift_m"] = round(dz, 2)
            meta["spawn_z_m"] = round(float(pos[0][2]), 2)

    payload = {
        "protocol_version": "phase3_outdoor_complex_v0",
        "scene": "outdoor_complex",
        "n_episodes": len(tagged),
        "episodes": tagged,
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d outdoor_complex episodes)", out, len(tagged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
