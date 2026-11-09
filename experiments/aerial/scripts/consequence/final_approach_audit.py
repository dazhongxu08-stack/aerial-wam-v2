"""Why timeouts on clean routes stop short: per-step log analysis after the drone first gets within --near-m.

For each timed-out flight: closest approach, final distance, share of steps with no candidate passing the
survival gate, chosen families, and predicted minus realised 10-step goal progress for the chosen action.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--art", default="experiments/aerial/rl/artifacts")
    ap.add_argument("--routes", required=True)
    ap.add_argument("--near-m", type=float, default=25.0)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    keep = set(args.routes.split(","))
    fam_near = collections.Counter()
    fam_far = collections.Counter()
    summary = collections.defaultdict(list)
    for pat in args.runs:
        for d in sorted(glob.glob(f"{args.art}/{pat}")):
            for e in json.load(open(f"{d}/MPC_EVAL.json"))["episodes"]:
                if "arrived" not in e or e["route_id"] not in keep or e["end_reason"] != "max_steps":
                    continue
                st = json.load(open(f"{d}/steps_{e['route_id']}.json"))
                dg = np.array([s["d_goal"] for s in st])
                npass = np.array([s["n_pass"] for s in st])
                near = np.where(dg < args.near_m)[0]
                tag = f"{d.split('/')[-1]} {e['route_id']}"
                if not len(near):
                    print(tag, f"never within {args.near_m:.0f} m, min {dg.min():.1f}")
                    summary["never_near"].append(tag)
                    continue
                i0 = int(near[0])
                fam_far.update(s["family"] for s in st[:i0])
                fam_near.update(s["family"] for s in st[i0:])
                err = [st[k]["pred_prog"] - (dg[k] - dg[k + 10]) for k in range(i0, len(st) - 10)]
                no_pass = float(np.mean(npass[i0:] == 0))
                surv = np.array([s["surv"] for s in st[i0:]])
                print(tag, f"near_at {i0}/{len(st)} dmin {dg.min():.1f} final {dg[-1]:.1f} "
                      f"no_pass {no_pass:.2f} chosen_surv_med {np.median(surv):.2f} "
                      f"pred_minus_real_10step {np.mean(err):+.2f} m "
                      f"families {collections.Counter(s['family'] for s in st[i0:]).most_common(4)}")
                summary["no_pass"].append(no_pass)
                summary["err"].append(float(np.mean(err)) if err else 0.0)
                summary["dmin"].append(float(dg.min()))
                summary["final"].append(float(dg[-1]))
    n = len(summary["dmin"])
    print(f"\nflights near goal {n}, never near {len(summary['never_near'])}")
    if n:
        print(f"median closest {np.median(summary['dmin']):.1f} m, median final {np.median(summary['final']):.1f} m, "
              f"median no-pass share {np.median(summary['no_pass']):.2f}, mean pred-minus-real {np.mean(summary['err']):+.2f} m")
    tot_n, tot_f = sum(fam_near.values()), sum(fam_far.values())
    print("families near goal", {k: round(v / tot_n, 2) for k, v in fam_near.most_common()})
    print("families before   ", {k: round(v / tot_f, 2) for k, v in fam_far.most_common()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
