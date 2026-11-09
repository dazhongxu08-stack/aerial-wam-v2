"""Offline audit of v8 vs v9 on per-candidate branch outcomes.

Reads eval_consequence_gate rows (one file per checkpoint seed) and reports, per probe set:
calibration of the ensemble survival against the simulator outcome, how often a safe
candidate with forward progress exists at all, and how mean / min / mean-k*std ensemble
survival would pick under the frozen rule (surv >= 0.95, then highest predicted progress).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts")
SURV_MIN = 0.95


def load(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    runs = [r["rows"] if isinstance(r, dict) else r for r in runs]
    out = []
    for rs in zip(*runs):
        s = np.array([r["surv"] for r in rs])
        p = np.array([r["pred_prog"] for r in rs]).mean(0)
        r0 = rs[0]
        out.append({"probe": r0["probe"], "surv": s, "pred_prog": p, "safe": np.array(r0["safe"]),
                    "allowed": np.array(r0["allowed"]),
                    "progress": np.array([np.nan if v is None else v for v in r0["progress"]], float)})
    return out


def pick(surv, prog, allowed):
    ok = allowed & (surv >= SURV_MIN)
    if ok.any():
        return int(np.where(ok)[0][np.argmax(prog[ok])])
    cand = np.where(allowed)[0] if allowed.any() else np.arange(len(surv))
    return int(cand[np.argmax(surv[cand])])


def report(name, rows, why=None):
    if why is not None:
        rows = [r for r in rows if why(r["probe"])]
    if not rows:
        return
    mean = np.concatenate([r["surv"].mean(0) for r in rows])
    safe = np.concatenate([r["safe"] for r in rows])
    print(f"\n== {name}: probes={len(rows)} candidates={len(safe)} actual_safe={safe.mean():.3f}")
    for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 0.98), (0.98, 1.01)]:
        m = (mean >= lo) & (mean < hi)
        if m.any():
            print(f"  pred surv [{lo:.2f},{hi:.2f}): n={m.sum():4d} actual safe={safe[m].mean():.3f}")
    good = [np.any(r["safe"] & r["allowed"] & (np.nan_to_num(r["progress"], nan=-9) > 0.5)) for r in rows]
    print(f"  states with a safe allowed candidate gaining >0.5 m: {np.mean(good):.3f}")
    for rule, f in [("mean", lambda s: s.mean(0)), ("min", lambda s: s.min(0)),
                    ("mean-1std", lambda s: s.mean(0) - s.std(0))]:
        picks = [pick(f(r["surv"]), r["pred_prog"], r["allowed"]) for r in rows]
        ps = np.mean([r["safe"][k] for r, k in zip(rows, picks)])
        pg = np.nanmean([r["progress"][k] if r["safe"][k] else np.nan for r, k in zip(rows, picks)])
        print(f"  pick rule {rule:9s}: picked safe={ps:.3f} mean progress of safe picks={pg:.2f} m")


def main():
    sets = json.loads(Path(sys.argv[1]).read_text())
    for set_name, spec in sets.items():
        why = None
        if "why_file" in spec:
            rev = json.loads(Path(spec["why_file"]).read_text())["probes"]
            kinds = {p["probe"]: p["why"].split("_")[0] for p in rev}
            for kind in ("lowclear", "before", "stuck", "random"):
                for model, files in spec["models"].items():
                    report(f"{set_name} [{kind}] {model}", load(files), lambda q, k=kind: kinds.get(q) == k)
        for model, files in spec["models"].items():
            report(f"{set_name} [all] {model}", load(files), why)


if __name__ == "__main__":
    main()
