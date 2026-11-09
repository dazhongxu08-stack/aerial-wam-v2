#!/usr/bin/env python3
"""Fine-tune the DA3 decoder on every collected pair. Encoder stays frozen.

There is no holdout in this run. The kept step is the one with the smallest
training obstacle-band cone error. The decoder is encrypted beside the pairs.
The live depth weight is not replaced, and no plaintext checkpoint is left
on disk.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

os.environ.setdefault("AERIAL_DA3_FP16", "1")

ROOT = Path(os.environ.get("YHS_PAIRS", "/home/user/rt/da3_pairs"))
WEIGHTS = os.environ.get("RT_RAM", "/dev/shm/rtw") + "/weights/m_d.bin"
STEPS = 480
ENC_OUT = ROOT / "m_d_office_dec.pt.enc"


def _load(name: str):
    z = np.load(ROOT / name)
    rgb = np.ascontiguousarray(z["rgb"])
    depth = z["depth"].astype(np.float32)
    seen = z["seen"].astype(bool)
    pose = [round(float(v), 3) for v in z["pose"]]
    z.close()
    valid = seen & np.isfinite(depth) & (depth > 0.3) & (depth < 3.0)
    return {"name": name, "rgb": rgb, "depth": depth, "valid": valid, "pose": pose}


def _bins(real, pred, mask):
    out = []
    for lo, hi in ((0.3, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)):
        m = mask & (real >= lo) & (real < hi) & np.isfinite(pred) & (pred > 0.05)
        if int(m.sum()) < 30:
            out.append({"real_m": [lo, hi], "n": int(m.sum())})
            continue
        err = np.abs(pred[m] - real[m])
        out.append({
            "real_m": [lo, hi],
            "n": int(m.sum()),
            "med_abs_m": round(float(np.median(err)), 3),
            "p90_abs_m": round(float(np.percentile(err, 90)), 3),
        })
    return out


def _cone_score(rows) -> float:
    err = []
    for row in rows:
        for pred, real in zip(row["cones_model"], row["cones_real"]):
            if pred is None or real is None:
                continue
            err.append(abs(pred - real))
    return 1e9 if not err else float(sum(err) / len(err))


def _score(items):
    num, den = 0.0, 0
    for real, pred, mask in items:
        m = mask & (real >= 0.6) & (real < 2.0) & np.isfinite(pred) & (pred > 0.05)
        if int(m.sum()) < 30:
            continue
        num += float(np.median(np.abs(pred[m] - real[m])))
        den += 1
    return 1e9 if den == 0 else num / den


def _detach(feats):
    import torch
    if torch.is_tensor(feats):
        return feats.detach()
    if isinstance(feats, dict):
        return {k: _detach(v) for k, v in feats.items()}
    if isinstance(feats, (list, tuple)):
        return type(feats)(_detach(v) for v in feats)
    return feats


def _predict(model, feats):
    import torch
    out = model.decoder(feats, model.image_size, model.image_size, patch_start_idx=0)
    depth = out["depth"][:, 0]
    if depth.shape[-2:] != (model.image_size, model.image_size):
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(model.image_size, model.image_size),
            mode="bilinear", align_corners=False)[:, 0]
    return depth[0]


def _cones(depth, valid):
    from aerial_finder.nav.consequence_model import pool_min
    from pinhole90 import band_mins, ground_guard_rows
    rows = ground_guard_rows(386.394, 243.503, 480)
    masked = np.where(valid & np.isfinite(depth) & (depth > 0.05), depth, np.inf)
    vals = band_mins(pool_min(masked.astype(np.float32)), *rows)
    return [None if not np.isfinite(v) else round(float(v), 2) for v in vals]


def _encode_all(model, frames):
    import torch
    encoded = []
    for frame in frames:
        rgb = torch.from_numpy(frame["rgb"]).view(1, 1, *frame["rgb"].shape).to("cuda")
        with torch.no_grad():
            feats, _aux = model.encoder(
                model._prep_last_frame(rgb).unsqueeze(1), cam_token=None, export_feat_layers=[])
        encoded.append(_detach(feats))
    return encoded


def _eval(model, frames, feats):
    import torch
    rows = []
    score_items = []
    model.decoder.eval()
    with torch.no_grad():
        for frame, feat in zip(frames, feats):
            pred = _predict(model, feat).clamp(min=0.05).float().cpu().numpy()
            rows.append({
                "file": frame["name"],
                "pose": frame["pose"],
                "bins": _bins(frame["depth"], pred, frame["valid"]),
                "cones_model": _cones(pred, frame["valid"]),
                "cones_real": _cones(frame["depth"], frame["valid"]),
            })
            score_items.append((frame["depth"], pred, frame["valid"]))
    return rows, _score(score_items)


def main() -> None:
    import torch
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor

    names = tuple(path.name for path in sorted(ROOT.glob("pair_*.npz")))
    train = [_load(name) for name in names]
    pred = DepthMinPredictor.from_checkpoint(WEIGHTS, device="cuda")
    pred.forward_scale = 1.0
    pred.forward_exponent = 1.0
    model = pred._model
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    train_feats = _encode_all(model, train)
    model.encoder.cpu()
    torch.cuda.empty_cache()
    gts = [torch.from_numpy(f["depth"]).to("cuda") for f in train]
    masks = [torch.from_numpy(f["valid"]).to("cuda") for f in train]
    before, base_pixels = _eval(model, train, train_feats)
    best = _cone_score(before)
    best_step = 0
    best_decoder = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    opt = torch.optim.AdamW(
        [p for p in model.decoder.parameters() if p.requires_grad], lr=1e-5, weight_decay=1e-2)
    for step in range(1, STEPS + 1):
        i = (step - 1) % len(train)
        model.decoder.train()
        depth = _predict(model, train_feats[i]).clamp(min=0.05)
        loss = torch.nn.functional.smooth_l1_loss(
            torch.log(depth[masks[i]]), torch.log(gts[i][masks[i]].clamp(min=0.05)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 60 == 0:
            rows, _pixels = _eval(model, train, train_feats)
            score = _cone_score(rows)
            print(json.dumps({"step": step, "train_cone_mae_m": round(score, 3)}), flush=True)
            if score < best:
                best = score
                best_step = step
                best_decoder = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    model.decoder.load_state_dict(best_decoder)
    model.decoder.to("cuda").eval()
    train_after, train_pixels = _eval(model, train, train_feats)
    plain = Path("/dev/shm/rtw/m_d_office_dec.pt")
    plain.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "da3_decoder", "step": best_step, "decoder": best_decoder}, plain)
    try:
        subprocess.check_call([
            "openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-salt",
            "-in", str(plain), "-out", str(ENC_OUT),
            "-pass", "file:" + str(Path.home() / ".config/rt/k"),
        ])
    finally:
        plain.unlink(missing_ok=True)
    os.chmod(ENC_OUT, 0o600)
    cones = [{
        "file": row["file"],
        "model": row["cones_model"],
        "real": row["cones_real"],
    } for row in train_after]
    report = {
        "installed": False,
        "encrypted": str(ENC_OUT),
        "live_weight_replaced": False,
        "train_frames": len(train),
        "best_step": best_step,
        "train_cone_mae_before_m": round(_cone_score(before), 3),
        "train_cone_mae_after_m": round(_cone_score(train_after), 3),
        "train_pixel_before_m": round(base_pixels, 3),
        "train_pixel_after_m": round(train_pixels, 3),
        "cones": cones,
    }
    (ROOT / "finetune_all33.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in report if k != "cones"}, ensure_ascii=False), flush=True)
    for row in cones:
        if row["file"] in {"pair_23.npz", "pair_24.npz", "pair_32.npz"}:
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
