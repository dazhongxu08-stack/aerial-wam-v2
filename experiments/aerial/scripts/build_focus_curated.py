#!/usr/bin/env python3
"""Copy curated episodes for a route subset into a new FT dataset."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--routes", required=True, help="comma-separated route_idx")
    args = p.parse_args()
    src = _REPO / args.src
    dst = _REPO / args.dst
    keep_r = {int(x) for x in args.routes.split(",") if x.strip()}
    manifest = json.loads((src / "manifest.json").read_text())
    eps = [e for e in manifest.get("episodes", []) if int(e["route_idx"]) in keep_r]
    dst.mkdir(parents=True, exist_ok=True)
    out_eps = []
    for i, e in enumerate(eps):
        src_f = src / e["file"]
        dst_f = dst / f"episode_{i:05d}.npz"
        shutil.copy2(src_f, dst_f)
        out_eps.append({**e, "file": dst_f.name})
    payload = {**manifest, "episodes": out_eps, "meta": {**manifest.get("meta", {}), "focus_routes": sorted(keep_r)}}
    (dst / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[focus-curated] routes={sorted(keep_r)} n={len(out_eps)} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
