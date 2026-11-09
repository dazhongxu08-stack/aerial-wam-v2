#!/usr/bin/env python3
"""G0 gate on the 20 holdout routes.

Every step flies forward. Even steps also yaw. The world model sees RGB and
the 10-D flight proprio only. Depth and pose build the action-integrated
label and never enter the network. The prior uses the argmax code, not the
sample from ``step()``.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from experiments.aerial.rl.occ_grid import (
    NZ,
    NY,
    NX,
    integrate_label,
    labels_for_episode,
    score_gate_window,
)
from experiments.aerial.rl.occ_topo_loss import HORIZON, OccDecoder

logger = logging.getLogger("eval_occ_topo_g0")

DX = 0.6
DYAW = 0.1
HISTORY = 16
MAX_STEPS = 96


def gate_action(step_index: int, yaw_sign: float) -> np.ndarray:
    """Forward on every step. Even steps also yaw. ``yaw_sign`` picks the side."""
    action = np.array([DX, 0.0, 0.0, 0.0], dtype=np.float64)
    if int(step_index) % 2 == 0:
        action[3] = float(yaw_sign) * DYAW
    return action


def mode_code(probs: torch.Tensor) -> torch.Tensor:
    index = probs.argmax(dim=-1)
    one = F.one_hot(index, probs.shape[-1]).to(dtype=probs.dtype)
    return one.reshape(probs.shape[0], -1)


def load_model(ckpt: Path, device: str):
    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics

    model = TorchRSSMDynamics(proprio_dim=10, device=device)
    payload = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    decoder = OccDecoder(in_dim=int(model.latent_dim)).to(model.device, model.torch_dtype)
    decoder.load_state_dict(payload["occ_decoder"])
    model.eval()
    decoder.eval()
    return model, decoder


def score_episode(path: Path, model, decoder, stride: int) -> Dict[str, Any]:
    from experiments.aerial.rl.train_occ_topo import decode_rgb

    with np.load(path) as z:
        blob = np.asarray(z["jpeg"])
        offsets = np.asarray(z["jpeg_offsets"])
        proprio = np.asarray(z["proprio"], dtype=np.float32)
        action = np.asarray(z["action"], dtype=np.float64)
        depth = np.asarray(z["depth"])
        pose = np.asarray(z["label_state"])
    if proprio.shape[-1] != 10:
        raise RuntimeError("%s proprio width %s" % (path, proprio.shape[-1]))
    grids = labels_for_episode(depth, pose, stride=stride)
    length = int(proprio.shape[0])
    h = model.rssm.initial_h(1, model.device, model.torch_dtype)
    open_flags: List[Optional[bool]] = []
    forward_flags: List[Optional[bool]] = []
    errors: List[float] = []
    last = length - HORIZON - 1
    with torch.no_grad():
        for t in range(length):
            rgb = torch.from_numpy(decode_rgb(blob, offsets, t)).unsqueeze(0).to(model.device)
            prop = torch.from_numpy(proprio[t : t + 1]).to(model.device, model.torch_dtype)
            embed = model._embed(rgb, prop)
            z = mode_code(model.rssm.post_probs(h, embed))
            if HISTORY - 1 <= t <= last and (t - (HISTORY - 1)) % HORIZON == 0:
                acts = action[t : t + HORIZON]
                label8 = integrate_label(grids[t], acts)
                h_k, z_k = h, z
                for k in range(HORIZON):
                    a = torch.from_numpy(acts[k].astype(np.float32)).reshape(1, 4).to(
                        model.device, model.torch_dtype
                    )
                    h_k = model.rssm.advance_h(h_k, z_k, a)
                    z_k = mode_code(model.rssm.prior_probs(h_k))
                logits = decoder(torch.cat([h_k, z_k], dim=-1))
                pred = logits.argmax(dim=-1).reshape(NZ, NY, NX).cpu().numpy().astype(np.uint8)
                window = score_gate_window(grids[t], label8, pred, float(acts[:, 3].sum()))
                open_flags.append(window["open"])
                forward_flags.append(window["forward"])
                errors.extend(window["errors"])
            if t + 1 < length:
                a = torch.from_numpy(action[t].astype(np.float32)).reshape(1, 4).to(
                    model.device, model.torch_dtype
                )
                h = model.rssm.advance_h(h, z, a)
    return {"open": open_flags, "forward": forward_flags, "errors": errors, "frames": length}


def _rate(flags: List[Optional[bool]]) -> Dict[str, Any]:
    counted = [bool(f) for f in flags if f is not None]
    return {
        "counted": len(counted),
        "passed": int(sum(counted)),
        "rate": (sum(counted) / len(counted)) if counted else None,
    }


def fly_route(env, route: Dict[str, Any], yaw_sign: float, max_steps: int, spawn_z: float, clear_m: float):
    from experiments.aerial.rl.collect_occ_topo import EpisodeBuffer
    from experiments.aerial.rl.spawn_utils import reset_with_spawn_retries

    pos = [list(map(float, p)) for p in route["pos"]]
    yaw0 = float(route["yaw"][0])
    ep = {"pos": pos, "yaw": [yaw0] * len(route["yaw"]), "gpt_instruction": route.get("gpt_instruction", "")}
    used, obs, _err, failed = reset_with_spawn_retries(
        env, ep,
        min_spawn_z=float(spawn_z),
        spawn_z_retry_m=5.0,
        spawn_z_max_retries=4,
        spawn_tol_m=12.0,
        mock=False,
        min_spawn_clear_m=float(clear_m),
    )
    meta = {"route_idx": route.get("route_idx"), "yaw_sign": float(yaw_sign), "odom_source": "commanded_body_delta", "imu_source": "getImuData"}
    if failed or obs is None:
        meta["end_reason"] = "spawn_fail"
        return None, meta
    buf = EpisodeBuffer()
    buf.add(obs, np.zeros(4), gate_action(0, yaw_sign))
    end = "max_steps"
    for step_index in range(int(max_steps)):
        action = np.asarray(buf.action[-1], dtype=np.float64)
        obs, info = env.step(action)
        cmd = np.asarray(info.get("cmd", action), dtype=np.float64).reshape(4)
        nxt = gate_action(step_index + 1, yaw_sign) if step_index + 1 < int(max_steps) else np.zeros(4)
        buf.add(obs, cmd, nxt)
        if bool(obs.collided):
            end = "collision"
            break
    meta["end_reason"] = end
    meta["steps"] = len(buf) - 1
    return buf, meta


def main() -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json",
    )
    parser.add_argument(
        "--ckpt",
        default="experiments/aerial/rl/artifacts/occ_topo_g0_20261010/occ_topo_wm.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/aerial/rl/artifacts/occ_topo_g0_20261010/holdout_gate",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--min-spawn-z", type=float, default=24.0)
    parser.add_argument("--min-spawn-clear-m", type=float, default=2.0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    routes = json.loads(Path(args.holdout).read_text(encoding="utf-8"))["episodes"]
    routes = sorted(routes, key=lambda r: int(r.get("route_idx", 0)))
    routes = [r for r in routes if int(r.get("route_idx", 0)) <= 19]

    model, decoder = load_model(Path(args.ckpt), args.device)
    logger.info("loaded %s", args.ckpt)

    if not args.score_only:
        from experiments.aerial.rl.env.airsim_env import AirSimDroneEnv, AirSimEnvConfig

        cfg = AirSimEnvConfig(step_hz=5.0, grab_depth=True, health_check=True)
        with AirSimDroneEnv(cfg) as env:
            for route in routes:
                for yaw_sign in (1.0, -1.0):
                    name = "route%02d_yaw%s.npz" % (int(route["route_idx"]), "p" if yaw_sign > 0 else "n")
                    dest = out / name
                    if dest.is_file():
                        logger.info("have %s", name)
                        continue
                    buf, meta = fly_route(
                        env, route, yaw_sign, int(args.max_steps),
                        float(args.min_spawn_z), float(args.min_spawn_clear_m),
                    )
                    if buf is None:
                        logger.info("spawn fail %s", name)
                        continue
                    buf.save(dest, meta)
                    logger.info("flew %s steps=%s end=%s", name, meta.get("steps"), meta.get("end_reason"))

    open_flags: List[Optional[bool]] = []
    forward_flags: List[Optional[bool]] = []
    errors: List[float] = []
    for path in sorted(out.glob("route*.npz")):
        part = score_episode(path, model, decoder, int(args.stride))
        open_flags.extend(part["open"])
        forward_flags.extend(part["forward"])
        errors.extend(part["errors"])
        logger.info(
            "scored %s frames=%s open=%s forward=%s",
            path.name,
            part["frames"],
            _rate(part["open"]),
            _rate(part["forward"]),
        )
    open_stat = _rate(open_flags)
    forward_stat = _rate(forward_flags)
    ray_median = None
    if errors:
        ordered = sorted(errors)
        ray_median = float(ordered[len(ordered) // 2])
    from experiments.aerial.rl.occ_grid import gate_passes

    open_ok = gate_passes(open_flags)
    forward_ok = gate_passes(forward_flags)
    ray_ok = ray_median is not None and ray_median <= 2.0
    report = {
        "open": open_stat,
        "forward": forward_stat,
        "ray_median_m": ray_median,
        "ray_count": len(errors),
        "open_pass": open_ok,
        "forward_pass": forward_ok,
        "ray_pass": ray_ok,
        "pass": bool(open_ok and forward_ok and ray_ok),
    }
    (out / "G0_GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("gate %s", json.dumps(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
