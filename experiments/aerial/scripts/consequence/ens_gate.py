"""Ensemble gate from per-model eval rows: mean survival / mean predicted progress,
then the same restricted metrics and gate as eval_consequence_gate (--select filter)."""
import json, sys
import numpy as np

tau = 0.7
gs = [json.load(open(p)) for p in sys.argv[1:]]
ids = set.intersection(*[{r["probe"] for r in g["rows"]} for g in gs])
by = [{r["probe"]: r for r in g["rows"]} for g in gs]
n = top_s = top_p = pairs = corr = 0
rs_sum = rp_sum = orc = 0.0
picks = {}
for pid in sorted(ids):
    rs = [b[pid] for b in by]
    r = rs[0]
    s = np.mean([x["surv"] for x in rs], 0)
    p = np.mean([x["pred_prog"] for x in rs], 0)
    keep = np.asarray(r["allowed"]); safe = np.asarray(r["safe"])
    if safe[keep].all() or (~safe[keep]).all():
        continue
    prog = np.asarray([np.nan if v is None else v for v in r["progress"]], np.float64)
    n += 1
    ok = keep & (s >= tau)
    c = int(np.argmax(np.where(ok, p, -np.inf))) if ok.any() else int(np.argmax(np.where(keep, s, -np.inf)))
    picks[r["families"][c]] = picks.get(r["families"][c], 0) + 1
    top_s += int(safe[c]); top_p += int(safe[c] and prog[c] > 0.5)
    for i in np.flatnonzero(safe & keep):
        for j in np.flatnonzero(~safe & keep):
            pairs += 1; corr += int(s[i] > s[j])
    rs_sum += float(safe[keep].mean())
    rp_sum += float(np.mean(safe[keep] & (np.nan_to_num(prog[keep], nan=-1.0) > 0.5)))
    o = int(np.nanargmax(np.where(safe & keep, prog, -1e9)))
    orc += int(safe[o] and prog[o] > 0.5)
m = {"pairwise": corr / pairs, "top1_safe": top_s / n, "top1_progress": top_p / n}
rnd = {"top1_safe": rs_sum / n, "top1_progress": rp_sum / n}
checks = {"pairwise>=0.7": m["pairwise"] >= 0.7, "top1_safe>=0.7": m["top1_safe"] >= 0.7,
          "top1_safe>=random": m["top1_safe"] >= rnd["top1_safe"],
          "top1_progress>=random+0.15": m["top1_progress"] >= rnd["top1_progress"] + 0.15}
print("probes=%d mixed=%d pairwise=%.3f top1_safe=%.3f (rand %.3f) top1_prog=%.3f (rand %.3f, need %.3f, oracle %.3f) gate=%s %s"
      % (len(ids), n, m["pairwise"], m["top1_safe"], rnd["top1_safe"], m["top1_progress"], rnd["top1_progress"],
         rnd["top1_progress"] + 0.15, orc / n, all(checks.values()), picks))
print("checks", checks)
