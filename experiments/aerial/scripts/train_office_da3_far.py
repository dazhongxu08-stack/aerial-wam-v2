#!/usr/bin/env python3
"""DA3 far/near contrast on the horizon-capped p20 cone.

Init from lowbox. Does not overwrite that checkpoint. No consequence, no robot.
Open-floor DA is no longer capped at 3.2 m.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_office_train as col  # noqa: E402
from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402
from office_bands import cones3_forward, horizon_row_56  # noqa: E402

MIX = Path("/data/simworld/run/indoor_office_mix")
ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
CKPT = ART / "da3_office_lowbox_20261103/depth_office_cone_lowbox.pt"
OUT = ART / "da3_office_far_20261103/depth_office_cone_far.pt"
FAR = 30.0
HZ56 = horizon_row_56()
NEAR_GT = 1.5
FAR_GT = 5.0
AISLE_DA_MIN = 4.0
WALL_DA_MAX = 1.5


def decode_one(z, index: int) -> np.ndarray:
    import cv2

    off = z["jpeg_offsets"]
    blob = z["jpeg"]
    return np.ascontiguousarray(
        cv2.imdecode(np.frombuffer(blob[int(off[index]) : int(off[index + 1])], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    )


def pool_min_t(depth: torch.Tensor, out: int = 56) -> torch.Tensor:
    b, h, w = depth.shape
    fh, fw = h // out, w // out
    d = depth[:, : fh * out, : fw * out].reshape(b, out, fh, out, fw)
    return d.amin(dim=(2, 4))


def hide_pool(pred: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    seen = rgb.sum(dim=-1) > 12
    depth = torch.where(seen, pred, torch.full_like(pred, FAR))
    depth = depth.clone()
    depth[:, col.HIDE_ROW :, :] = FAR
    return pool_min_t(depth)


def forward_maps(model, rgb: np.ndarray) -> torch.Tensor:
    window = torch.from_numpy(np.ascontiguousarray(rgb)).to("cuda").unsqueeze(1)
    pred, _ = model.predict_from_window(window)
    return hide_pool(pred, window[:, 0])


def p20_fwd_t(d56: torch.Tensor) -> torch.Tensor:
    h, w = d56.shape[-2:]
    hi = min(int(HZ56), h - 1)
    lo = max(h // 8, 0)
    if lo >= hi:
        lo = max(0, hi - 4)
    band = d56[:, lo:hi, w // 4 : w - w // 4]
    flat = band.reshape(band.shape[0], -1)
    k = max(1, int(round(0.20 * (flat.shape[1] - 1))))
    return flat.kthvalue(k + 1, dim=-1).values


def far_near_loss(p56: torch.Tensor, gt56: torch.Tensor) -> torch.Tensor:
    pf = p20_fwd_t(p56).clamp(0.05, FAR)
    gf = p20_fwd_t(gt56).clamp(0.05, FAR)
    lp, lg = torch.log(pf), torch.log(gf)
    l1 = (lp - lg).abs()
    too_short = torch.relu(math.log(FAR_GT) - lp)
    too_long = torch.relu(lp - lg)
    far = (gf >= FAR_GT).float()
    near = (gf <= NEAR_GT).float()
    cone = l1 + 8.0 * far * too_short + 6.0 * near * too_long
    h, w = p56.shape[-2:]
    hi = min(int(HZ56), h - 1)
    lo = max(h // 8, 0)
    p = p56[:, lo:hi].clamp(0.05, FAR)
    g = gt56[:, lo:hi]
    far_px = torch.isfinite(g) & (g >= FAR_GT)
    near_px = torch.isfinite(g) & (g > 0.05) & (g <= NEAR_GT)
    pix_far = torch.relu(math.log(4.0) - torch.log(p))
    pix_near = torch.relu(torch.log(p) - torch.log(g.clamp(min=0.05)))
    n_far = far_px.flatten(1).sum(1).clamp(min=1)
    n_near = near_px.flatten(1).sum(1).clamp(min=1)
    extra = torch.zeros((), device=p56.device)
    if bool(far_px.any()):
        extra = extra + ((pix_far * far_px).flatten(1).sum(1) / n_far)[far_px.flatten(1).any(1)].mean()
    if bool(near_px.any()):
        extra = extra + ((pix_near * near_px).flatten(1).sum(1) / n_near)[near_px.flatten(1).any(1)].mean()
    return cone.mean() + 4.0 * extra


def index_mix() -> tuple[list, list]:
    review = json.loads((MIX / "AUTO_REVIEW.json").read_text())
    train, hold = [], []
    for row in review["episodes"]:
        if not row.get("keep"):
            continue
        path = MIX / "episodes" / row["file"]
        with np.load(path) as archive:
            depth = np.asarray(archive["depth56"], np.float32)
        bucket = hold if row.get("route_id") in ("office_val", "office_yaw_val") else train
        for i in range(len(depth)):
            fwd = float(cones3_forward(depth[i])[0])
            bucket.append((row["file"], i, fwd))
    return train, hold


class Loader:
    def __init__(self) -> None:
        self._name = ""
        self._ep = None

    def close(self) -> None:
        if self._ep is not None:
            self._ep.close()
            self._ep = None
            self._name = ""

    def batch(self, refs: list) -> tuple[np.ndarray, np.ndarray]:
        ordered = sorted(refs, key=lambda r: (r[0], r[1]))
        rgb, depth = [], []
        for name, index, _fwd in ordered:
            path = str(MIX / "episodes" / name)
            if self._name != path:
                self.close()
                self._ep = np.load(path)
                self._name = path
            rgb.append(decode_one(self._ep, index))
            depth.append(np.asarray(self._ep["depth56"][index], np.float32))
        self.close()
        return np.stack(rgb), np.stack(depth)


@torch.no_grad()
def eval_refs(model, refs: list, loader: Loader, limit: int = 0) -> dict:
    model.eval()
    use = refs
    if limit and len(refs) > limit:
        rng = np.random.default_rng(1)
        far = [r for r in refs if r[2] >= FAR_GT]
        near = [r for r in refs if r[2] <= NEAR_GT]
        rest = [r for r in refs if NEAR_GT < r[2] < FAR_GT]
        extra = []
        if rest:
            idx = rng.choice(len(rest), size=min(len(rest), 80), replace=False)
            extra = [rest[int(i)] for i in np.atleast_1d(idx)]
        use = far[:80] + near[:80] + extra
    rows = []
    ordered = sorted(use, key=lambda r: (r[0], r[1]))
    for i in range(0, len(ordered), 8):
        chunk = ordered[i : i + 8]
        rgb, gt = loader.batch(chunk)
        pred = forward_maps(model, rgb).detach().float().cpu().numpy()
        for j, ref in enumerate(chunk):
            rows.append((float(ref[2]), float(cones3_forward(pred[j])[0]), float(cones3_forward(gt[j])[0])))
    da_far = [d for g, d, _ in rows if g >= FAR_GT]
    da_near = [d for g, d, _ in rows if g <= NEAR_GT]
    gt_far = [g for g, d, _ in rows if g >= FAR_GT]
    gt_near = [g for g, d, _ in rows if g <= NEAR_GT]
    stats = {
        "n": len(rows),
        "far_n": len(da_far),
        "near_n": len(da_near),
        "far_gt": float(np.median(gt_far)) if gt_far else 0.0,
        "far_da": float(np.median(da_far)) if da_far else 0.0,
        "near_gt": float(np.median(gt_near)) if gt_near else 0.0,
        "near_da": float(np.median(da_near)) if da_near else 0.0,
    }
    print(
        f"  hold far n={stats['far_n']} gt {stats['far_gt']:.2f} da {stats['far_da']:.2f} | "
        f"near n={stats['near_n']} gt {stats['near_gt']:.2f} da {stats['near_da']:.2f}",
        flush=True,
    )
    return stats


def live_gate(ckpt_path: Path) -> dict:
    from experiments.aerial.rl.env.obs import Observation
    import collect_office_lowbox as boxes

    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    try:
        import apply_office_daylight as day

        day.apply(ucv, day.load_cfg())
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    boxes.save_restore(ucv)
    rows = json.loads((boxes.OUT / "lowbox_restore.json").read_text())
    for row in rows:
        n = row["name"]
        col.req(ucv, f"vset /object/{n}/object_mobility true")
        col.req(ucv, f"vset /object/{n}/physics false")
        col.req(ucv, f"vset /object/{n}/scale {row['scale']}")
        col.req(ucv, f"vset /object/{n}/rotation {row['rotation']}")
        col.req(ucv, f"vset /object/{n}/location {row['location']}")
    pred = DepthMinPredictor.from_checkpoint(str(ckpt_path), device="cuda")
    out = {}
    for name, pose in (
        ("aisle", (-480.0, -200.0, 0.0)),
        ("wall", (-520.0, -250.0, 0.0)),
        ("gap", (-160.0, -200.0, 0.0)),
    ):
        x, y, yaw = pose
        col.place(ucv, x, y, yaw)
        col.look(ucv, cam, x, y, yaw)
        packed = col.grab(ucv, cam)
        rgb, depth = packed
        image, seen, warped = col.warp_pair(rgb, depth)
        gt56 = col.pool_min(col.hide(warped, seen))
        gt = cones3_forward(gt56)
        pred.reset()
        da224 = np.asarray(
            pred.predict_depth(Observation(rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False)),
            np.float32,
        )
        if da224.shape[:2] != warped.shape[:2]:
            import cv2

            da224 = cv2.resize(da224, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
        da = cones3_forward(col.pool_min(col.hide(da224, seen)))
        rec = {"gt": [float(v) for v in gt], "da": [float(v) for v in da]}
        out[name] = rec
        print(f"LIVE {name} gt_f={rec['gt'][0]:.2f} da_f={rec['da'][0]:.2f}", flush=True)
    aisle_ok = out["aisle"]["da"][0] >= AISLE_DA_MIN
    wall_ok = out["wall"]["da"][0] <= WALL_DA_MAX
    gap_ok = out["gap"]["da"][0] >= 3.0 and out["gap"]["da"][0] > out["wall"]["da"][0] + 0.8
    out["ok"] = bool(aisle_ok and wall_ok and gap_ok)
    print(
        f"GATE live aisle {aisle_ok} wall {wall_ok} gap {gap_ok} -> {'PASS' if out['ok'] else 'FAIL'}",
        flush=True,
    )
    return out


def save_ckpt(model, src: dict, path: Path, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(src)
    payload["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload["office_far"] = extra
    torch.save(payload, str(path))
    print("SAVE", path, flush=True)


def main() -> int:
    print("CKPT", CKPT, "OUT", OUT, "HIDE", col.HIDE_ROW, "HZ56", HZ56, flush=True)
    train_refs, hold_refs = index_mix()
    far = [r for r in train_refs if r[2] >= FAR_GT]
    near = [r for r in train_refs if r[2] <= NEAR_GT]
    mid = [r for r in train_refs if NEAR_GT < r[2] < FAR_GT]
    print(f"index train {len(train_refs)} far {len(far)} near {len(near)} mid {len(mid)} hold {len(hold_refs)}", flush=True)
    loader = Loader()
    wrap = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
    model = wrap._model
    model.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    src = torch.load(str(CKPT), map_location="cpu")
    print("before hold", flush=True)
    before = eval_refs(model, hold_refs, loader, limit=240)
    print("before live (lowbox)", flush=True)
    live0 = live_gate(CKPT)
    opt = torch.optim.AdamW((p for p in model.decoder.parameters() if p.requires_grad), lr=3e-6)
    rng = np.random.default_rng(3)
    best = None
    best_state = None
    for step in range(1, 201):
        groups = []
        if far:
            groups.append([far[int(i)] for i in rng.integers(0, len(far), size=4)])
        if near:
            groups.append([near[int(i)] for i in rng.integers(0, len(near), size=4)])
        if mid:
            groups.append([mid[int(i)] for i in rng.integers(0, len(mid), size=2)])
        pick = [r for g in groups for r in g]
        rgb, gt56 = loader.batch(pick)
        p56 = forward_maps(model, rgb)
        gt_t = torch.as_tensor(gt56, dtype=torch.float32, device="cuda")
        loss = far_near_loss(p56, gt_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == 200:
            print(f"step {step} loss {float(loss):.4f}", flush=True)
            stats = eval_refs(model, hold_refs, loader, limit=240)
            score = max(0.0, FAR_GT - stats["far_da"]) + abs(stats["near_da"] - stats["near_gt"])
            save_ckpt(model, src, OUT.with_name("depth_office_cone_far_last.pt"), stats)
            if best is None or score < best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_ckpt(model, src, OUT, {**stats, "step": step, "score": score})
    if best_state is not None:
        model.load_state_dict(best_state)
    print("after hold", flush=True)
    after = eval_refs(model, hold_refs, loader, limit=0)
    print("after live", flush=True)
    live1 = live_gate(OUT)
    summary = {"before_hold": before, "after_hold": after, "before_live": live0, "after_live": live1}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "far_gate.json").write_text(json.dumps(summary, indent=2))
    print("DONE", "PASS" if live1.get("ok") else "FAIL", flush=True)
    return 0 if live1.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
