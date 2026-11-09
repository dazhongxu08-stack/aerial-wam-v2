"""Export the DA3 depth head as TorchScript for the standalone service.

Run from the training repo root (the only step that needs it):

    python experiments/aerial/deploy/aerial_finder/tools/export_da3.py \
        --ckpt experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt \
        --out weights/da3_depth_env.ts [--check-frames DIR_OF_JPGS]

The scripted module runs the ViT encoder once per frame and returns both the
calibrated depth (what ``DepthMinPredictor.predict_depth`` returns) and the
scene vector (what ``env_pair`` returns for the env-feature progress heads).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[5]


class Da3DepthEnv(nn.Module):
    def __init__(self, head, scale: float, exponent: float) -> None:
        super().__init__()
        self.head = head
        self.scale = float(scale)
        self.exponent = float(exponent)

    def forward(self, rgb: torch.Tensor):
        from experiments.aerial.rl.dynamics_torch import da3_env_vector

        h = self.head
        x = h._prep_last_frame(rgb)
        feats, _ = h.encoder(x.unsqueeze(1), cam_token=None, export_feat_layers=[])
        depth = h.decoder(feats, x.shape[-2], x.shape[-1], patch_start_idx=0)["depth"][:, 0]
        if depth.shape[-2:] != (h.image_size, h.image_size):
            depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(h.image_size, h.image_size),
                                                    mode="bilinear", align_corners=False)[:, 0]
        if self.scale != 1.0 or self.exponent != 1.0:
            depth = self.scale * torch.clamp(depth, min=0.0) ** self.exponent
        return depth, da3_env_vector(feats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", help="trace device; the export loads on any device")
    ap.add_argument("--check-frames", default="", help="dir of RGB jpg/png frames to compare against eager")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    import cv2

    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation
    from experiments.aerial.rl.precompute_env_consequence import env_pair

    pred = DepthMinPredictor.from_checkpoint(ROOT / args.ckpt, device=args.device)
    head = pred._model.eval()
    size = int(head.image_size)
    scale, exponent = getattr(pred, "forward_scale", 1.0), getattr(pred, "forward_exponent", 1.0)
    mod = Da3DepthEnv(head, scale, exponent).eval()
    ex = torch.randint(0, 255, (1, 1, size, size, 3), dtype=torch.uint8, device=args.device)
    with torch.no_grad():
        ts = torch.jit.trace(mod, ex, check_trace=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ts.save(str(out))
    meta = {"image_size": size, "n_frames": 1, "env_dim": None, "source_ckpt": args.ckpt,
            "forward_scale": scale, "forward_exponent": exponent}

    frames = []
    if args.check_frames:
        for p in sorted(Path(args.check_frames).glob("*"))[:16]:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(np.ascontiguousarray(img[..., ::-1]))
    if not frames:
        frames = [np.random.default_rng(i).integers(0, 255, (size, size, 3), dtype=np.uint8) for i in range(3)]
    loaded = torch.jit.load(str(out), map_location=args.device)
    worst_d, worst_e = 0.0, 0.0
    for rgb in frames:
        r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR) if rgb.shape[:2] != (size, size) else rgb
        pred.reset()
        ref_d = pred.predict_depth(Observation(rgb=r, state=np.zeros(7, np.float32)))
        ref_e = env_pair(head, [r])[0][0].astype(np.float32)
        with torch.no_grad():
            d, e = loaded(torch.from_numpy(r).unsqueeze(0).unsqueeze(0).to(args.device))
        d, e = d[0].cpu().numpy(), e[0].cpu().numpy()
        meta["env_dim"] = int(e.shape[-1])
        worst_d = max(worst_d, float(np.max(np.abs(d - ref_d) / np.maximum(ref_d, 0.5))))
        worst_e = max(worst_e, float(np.max(np.abs(e - ref_e)) / (np.abs(ref_e).max() + 1e-6)))
    meta.update({"check_frames": len(frames), "max_rel_depth_err": worst_d, "max_rel_env_err": worst_e})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return 0 if worst_d < 1e-3 and worst_e < 2e-3 else 2


if __name__ == "__main__":
    sys.exit(main())
