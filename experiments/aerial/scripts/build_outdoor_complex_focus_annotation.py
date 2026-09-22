#!/usr/bin/env python3
"""Subset outdoor_complex_only.json to a route list (default 0,1,3,4)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json",
    )
    p.add_argument(
        "--out",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134.json",
    )
    p.add_argument("--routes", default="0,1,3,4", help="comma-separated route_idx")
    args = p.parse_args()
    src = _REPO / args.src
    dst = _REPO / args.out
    keep = {int(x) for x in args.routes.split(",") if x.strip()}
    data = json.loads(src.read_text(encoding="utf-8"))
    eps = data.get("episodes", [])
    picked = [e for i, e in enumerate(eps) if i in keep]
    if len(picked) != len(keep):
        missing = keep - {i for i in range(len(eps)) if i in keep}
        print(f"WARN: wanted routes {sorted(keep)}, got {len(picked)} episodes", file=sys.stderr)
    out = dict(data)
    out["episodes"] = picked
    out["n_episodes"] = len(picked)
    out["focus_routes"] = sorted(keep)
    out["subset_of"] = str(src.relative_to(_REPO))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {len(picked)} episodes -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
