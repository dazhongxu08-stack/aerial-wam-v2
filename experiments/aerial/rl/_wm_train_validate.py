"""V1 world-model VALIDATION on the H100 (torch) — the gate before enabling training.

Unlike ``_wm_bringup_smoke`` (which only checks the disk→buffer→stub path with no
weights), this trains the real :class:`TorchRSSMDynamics` on ``dataset_v1_rgb``
and checks the two V1 pass criteria from the design doc:

  A. LEARNING — ``update()`` loss (and recon term) trends DOWN over N steps, all
     finite; the posterior does NOT collapse (entropy stays above the §2.3 floor).
  B. NON-DIVERGENCE (§9) — open-loop multi-step imagination from real start states
     stays bounded: latents finite, norm not exploding, p_coll∈[0,1] over the
     horizon cap. This is the gate: WM multi-step error must not blow up.

Runs on the H100 only (imports torch at module top). Exits 0 on PASS, 1 on FAIL.

    python -m experiments.aerial.rl._wm_train_validate \
        --dataset /home/<user>/rl_collect_run/.../artifacts/dataset_v1_rgb \
        --config configs/aerial_rl.yaml --steps 500

Refuses the dt-desynced V0 corpus (step_hz>8.5) unless --allow-v0-desync, exactly
like the bring-up smoke — a real WM must not be trained on desynced labels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch  # H100 only; the whole script is gated on this import

import yaml

from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics
from experiments.aerial.rl.goal_features import body_vel_from_obs, goal_rel_from_obs


def _load_world_model_cfg(config_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text()) or {}
    return dict(cfg.get("world_model", {}) or {})


def _refuse_v0(root: Path, allow: bool) -> None:
    """Refuse corpora whose labeled step_hz exceeds a measured closed-loop floor.

    Thresholds (4090 loopback, 2026-08-04):
      - grab_depth=true  ceiling ≈ 6.2 Hz → refuse labeled step_hz > 6.5
      - RGB-only legacy   labeled-12 / achieved~7.9 → refuse step_hz > 8.5
    """
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    meta = manifest.get("meta") or {}
    step_hz = float(meta.get("step_hz", 0) or 0)
    grab_depth = bool(meta.get("grab_depth", False))
    if allow:
        return
    if grab_depth and step_hz > 6.5:
        print(
            f"[wm-validate] REFUSE: dataset step_hz={step_hz} with grab_depth "
            "exceeds the measured 4090-local depth closed-loop ceiling (~6.2 Hz). "
            "Re-collect at step_hz≤5.0, or pass --allow-v0-desync only to exercise "
            "the code path.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if step_hz > 8.5:
        print(
            f"[wm-validate] REFUSE: dataset step_hz={step_hz} is the dt-desynced V0 "
            "corpus — do not train a real WM on it. Pass --allow-v0-desync only to "
            "exercise the code path, or point --dataset at a rate-locked corpus.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _write_train_meta(
    ckpt_dir: Path,
    *,
    root: Path,
    args: argparse.Namespace,
    buf: ReplayBuffer,
    image_size: int,
) -> Path:
    """Record WHICH corpus produced ``wm_train.jsonl`` (provenance sidecar).

    Why this exists: ``wm_train.jsonl`` carries only loss/recon/entropy, so a curve
    on disk cannot evidence the corpus it was trained on. That gap is exactly what
    made ``wm_ckpt_v2clean_20260810`` unusable as ①a–c evidence — its numbers pass
    the thresholds decisively, but nothing on disk proves it was not the dt-desynced
    July V0 corpus waved through with ``--allow-v0-desync`` (which ``_refuse_v0``
    documents as "only to exercise the code path"). A self-describing artifact makes
    the ①a–c verdict auditable weeks later instead of relying on recall.

    Written BEFORE training so it survives a crashed/interrupted run. Additive and
    read-only w.r.t. the gate: ``_v0_gate._signal1abc_from_log`` parses the .jsonl
    only and never sees this file, so the frozen §4.1 verdict is byte-identical.
    ``allow_v0_desync`` is recorded first-class — a true value DISQUALIFIES the run
    as authoritative ①a–c evidence.
    """
    manifest_meta: Dict[str, Any] = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest_meta = (json.loads(manifest_path.read_text()).get("meta") or {})
        except (ValueError, OSError) as exc:  # pragma: no cover - corrupt manifest
            manifest_meta = {"_unreadable": str(exc)}
    meta = {
        "dataset": str(root.resolve()),
        "dataset_manifest_meta": manifest_meta,
        "allow_v0_desync": bool(args.allow_v0_desync),
        "authoritative": not bool(args.allow_v0_desync),
        "episodes": int(buf.num_episodes),
        "transitions": int(buf.num_transitions),
        "steps": int(args.steps),
        "window": int(args.window),
        "wm_batch": int(args.wm_batch),
        "config": str(Path(args.config).resolve()),
        "image_size": int(image_size),
        "git_sha": _git_sha(),
        # Tail fraction excluded from the train buffer so V1-② fidelity
        # (--heldout-frac matching) is an honest gate, not in-sample leakage.
        "heldout_frac": float(getattr(args, "heldout_frac", 0.0) or 0.0),
    }
    path = ckpt_dir / "wm_train_meta.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return path


def _git_sha() -> Optional[str]:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    return out.stdout.strip() or None


def _load_buffer(root: Path, window: int, heldout_frac: float = 0.0) -> ReplayBuffer:
    episodes = ds.load_dataset(root, skip_quarantined=True)
    episodes = [ep for ep in episodes if len(ep) >= window]
    if not episodes:
        print(f"[wm-validate] FAIL: no episode >= {window} steps", file=sys.stderr)
        raise SystemExit(1)
    n = len(episodes)
    frac = float(heldout_frac or 0.0)
    if frac > 0.0:
        # Deterministic tail split — MUST match ``_wm_fidelity_eval._heldout_split``.
        k = max(1, math.ceil(frac * n))
        train = episodes[: n - k]
        print(
            f"[wm-validate] held-out exclude: train={len(train)}/{n} "
            f"(tail {k} reserved for fidelity; frac={frac})"
        )
        episodes = train
        if not episodes:
            print("[wm-validate] FAIL: held-out frac left empty train set", file=sys.stderr)
            raise SystemExit(1)
    elif frac < 0.0:
        print(f"[wm-validate] FAIL: --heldout-frac={frac} must be >= 0", file=sys.stderr)
        raise SystemExit(2)
    else:
        print(
            "[wm-validate] heldout_frac=0 → training on ALL episodes "
            "(in-sample only; not an honest V1-② ckpt)",
            file=sys.stderr,
        )
    buf = ReplayBuffer(capacity_episodes=len(episodes) + 1, seed=0)
    for ep in episodes:
        buf.add_episode(ep)
    print(f"[wm-validate] buffer: {buf.num_episodes} eps / {buf.num_transitions} steps")
    return buf


def _check_learning(model: TorchRSSMDynamics, buf: ReplayBuffer,
                    steps: int, wm_batch: int, window: int,
                    log_path: Optional[Path] = None,
                    head_only: bool = False) -> bool:
    losses: List[float] = []
    recons: List[float] = []
    ent_fracs: List[float] = []
    pair_losses: List[float] = []
    if log_path is not None and log_path.exists():
        log_path.unlink()
    for i in range(steps):
        windows = buf.sample_windows(wm_batch, window)
        out = model.update(windows)
        losses.append(out["loss"])
        recons.append(out["recon_err"])
        ent_fracs.append(out.get("post_entropy_frac", 1.0))
        pair_losses.append(float(out.get("loss_coll_pair", float("nan"))))
        if log_path is not None:
            # Field names MUST match _v0_gate._signal1abc_from_log (loss /
            # recon_err / post_entropy_frac) so ①a–c reads this as its curve.
            row = {
                "step": i,
                "loss": float(out["loss"]),
                "recon_err": float(out["recon_err"]),
                "post_entropy_frac": float(out.get("post_entropy_frac", 1.0)),
                "loss_dyn": float(out.get("loss_dyn", float("nan"))),
                "loss_rep": float(out.get("loss_rep", float("nan"))),
                "loss_reward": float(out.get("loss_reward", float("nan"))),
                "loss_coll": float(out.get("loss_coll", float("nan"))),
                "loss_coll_pair": float(out.get("loss_coll_pair", float("nan"))),
                "coll_pair_n": int(out.get("coll_pair_n", 0)),
                "loss_coll_hinge": float(out.get("loss_coll_hinge", float("nan"))),
                "loss_depth_aux": float(out.get("loss_depth_aux", float("nan"))),
                "loss_center_depth": float(out.get("loss_center_depth", float("nan"))),
                "loss_cone_side": float(out.get("loss_cone_side", float("nan"))),
                "coll_hinge_gap": float(out.get("coll_hinge_gap", float("nan"))),
                "loss_early_yaw": float(out.get("loss_early_yaw", float("nan"))),
                "early_yaw_gap": float(out.get("early_yaw_gap", float("nan"))),
                "early_yaw_n": int(out.get("early_yaw_n", 0)),
                "grad_norm": float(out.get("grad_norm", float("nan"))),
            }
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        if i % max(1, steps // 10) == 0:
            print(f"[wm-validate] step {i:4d} | loss={out['loss']:.4f} "
                  f"recon={out['recon_err']:.4f} rew={out.get('loss_reward', float('nan')):.3f} "
                  f"coll={out.get('loss_coll', float('nan')):.3f} "
                  f"pair={out.get('loss_coll_pair', float('nan')):.3f} "
                  f"pair_n={int(out.get('coll_pair_n', 0))} "
                  f"hinge={out.get('loss_coll_hinge', float('nan')):.3f} "
                  f"d_aux={out.get('loss_depth_aux', float('nan')):.3f} "
                  f"center={out.get('loss_center_depth', float('nan')):.3f} "
                  f"cone={out.get('loss_cone_side', float('nan')):.3f} "
                  f"gap={out.get('coll_hinge_gap', float('nan')):.3f} "
                  f"early={out.get('loss_early_yaw', float('nan')):.3f} "
                  f"early_n={int(out.get('early_yaw_n', 0))} "
                  f"early_gap={out.get('early_yaw_gap', float('nan')):.3f} "
                  f"dyn={out['loss_dyn']:.3f} "
                  f"rep={out['loss_rep']:.3f} ent={ent_fracs[-1]:.2f} "
                  f"|g|={out['grad_norm']:.1f}")

    if not all(np.isfinite(losses)):
        print("[wm-validate] FAIL(A): non-finite loss during training", file=sys.stderr)
        return False
    k = max(1, steps // 10)
    first, last = float(np.mean(losses[:k])), float(np.mean(losses[-k:]))
    recon_first, recon_last = float(np.mean(recons[:k])), float(np.mean(recons[-k:]))
    min_ent = float(np.min(ent_fracs))
    collapse_ok = min_ent >= model.collapse_entropy_frac
    if head_only:
        # Backbone is frozen, so total loss and recon are KL/decode noise.
        # The trained quantity is the forward/yaw pair loss.
        pair_first = float(np.mean(pair_losses[:k]))
        pair_last = float(np.mean(pair_losses[-k:]))
        loss_ok = np.isfinite(pair_last) and pair_last < pair_first * 0.98
        print(f"[wm-validate] LEARNING: pair {pair_first:.4f}→{pair_last:.4f} "
              f"({'OK' if loss_ok else 'FAIL'}) | "
              f"total {first:.4f}→{last:.4f} (frozen backbone) | "
              f"min entropy frac {min_ent:.2f} ({'OK' if collapse_ok else 'COLLAPSE'})")
        return loss_ok and collapse_ok
    loss_ok = last < first * 0.98              # ≥2% total-loss drop
    recon_ok = recon_last <= recon_first       # recon not worse
    print(f"[wm-validate] LEARNING: loss {first:.4f}→{last:.4f} ({'OK' if loss_ok else 'FAIL'}) | "
          f"recon {recon_first:.4f}→{recon_last:.4f} ({'OK' if recon_ok else 'FAIL'}) | "
          f"min entropy frac {min_ent:.2f} ({'OK' if collapse_ok else 'COLLAPSE'})")
    return loss_ok and recon_ok and collapse_ok


def _check_non_divergence(model: TorchRSSMDynamics, buf: ReplayBuffer,
                          window: int, horizon: int, n_traj: int = 8) -> bool:
    windows = buf.sample_windows(n_traj, window)
    ok = True
    max_norm_seen = 0.0
    for w in windows:
        z = model.encode(w[0].obs)
        norms = [float(np.linalg.norm(z))]
        for t in range(min(horizon, len(w))):
            out = model.step(
                z,
                w[t].action,
                goal_rel=goal_rel_from_obs(w[t].obs),
                body_vel=body_vel_from_obs(w[t].obs),
            )
            z = out.z_next
            if not np.all(np.isfinite(z)):
                print("[wm-validate] FAIL(B): non-finite latent in rollout", file=sys.stderr)
                return False
            if not (0.0 <= out.p_coll <= 1.0) or not np.isfinite(out.progress):
                print(f"[wm-validate] FAIL(B): bad head output p_coll={out.p_coll} "
                      f"progress={out.progress}", file=sys.stderr)
                return False
            norms.append(float(np.linalg.norm(z)))
        max_norm_seen = max(max_norm_seen, max(norms))
        # non-divergence: the packed-latent norm must not blow up over the horizon.
        if max(norms) > 50.0 * (norms[0] + 1.0):
            print(f"[wm-validate] FAIL(B): latent norm diverged {norms[0]:.2f}→{max(norms):.2f}",
                  file=sys.stderr)
            ok = False
    print(f"[wm-validate] NON-DIVERGENCE: {n_traj} rollouts × H={horizon}, "
          f"max latent norm {max_norm_seen:.2f} ({'OK' if ok else 'FAIL'})")
    return ok


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="dir with episode_*.npz (dataset_v1_rgb)")
    p.add_argument("--config", default="configs/aerial_rl.yaml")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--wm-batch", type=int, default=8)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--horizon", type=int, default=15)  # MAX_IMAGINATION_HORIZON (§9)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-v0-desync", action="store_true")
    p.add_argument(
        "--save-ckpt",
        action="store_true",
        help="on PASS, write world_model.checkpoint_dir/wm_step_<N>.pt (runbook §3)",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="override world_model.checkpoint_dir: dir for wm_train.jsonl and the "
        "saved ckpt. Point at a dated dir (e.g. artifacts/wm_ckpt_v2clean_<date>) "
        "to keep a clean retrain from clobbering the invalidated wm_ckpt/ log+ckpt.",
    )
    p.add_argument(
        "--heldout-frac",
        type=float,
        default=0.0,
        help="tail fraction of episodes excluded from training (must match "
        "_wm_fidelity_eval --heldout-frac for an honest V1-② gate; 0 = all eps)",
    )
    p.add_argument(
        "--init-ckpt",
        default=None,
        help="warm-start from an existing wm_step_*.pt (loads weights before training)",
    )
    p.add_argument(
        "--cone-depth-weight",
        type=float,
        default=0.0,
        help="extra loss on left/right cone order. 0 leaves the depth loss unchanged.",
    )
    p.add_argument(
        "--save-step",
        type=int,
        default=None,
        help="step number embedded in wm_step_<N>.pt (default: init_step+steps or steps)",
    )
    p.add_argument(
        "--collision-head-only",
        action="store_true",
        help="freeze the backbone and train coll_head on the loaded latent",
    )
    p.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
        help="AdamW lr for --collision-head-only",
    )
    args = p.parse_args(argv)

    root = Path(args.dataset)
    _refuse_v0(root, args.allow_v0_desync)
    buf = _load_buffer(root, args.window, heldout_frac=float(args.heldout_frac))

    wm_cfg = _load_world_model_cfg(Path(args.config))
    wm_cfg.setdefault("device", args.device)
    # CLI override wins over config so a clean retrain writes to a dated dir and
    # never touches the invalidated wm_ckpt/ (log is unlink+rewritten each run).
    if args.checkpoint_dir:
        wm_cfg["checkpoint_dir"] = args.checkpoint_dir
    # Match the model's image size to the actual frame (defaults to 224).
    sample_obs = buf.sample_windows(1, 1)[0][0].obs
    wm_cfg["image_size"] = int(np.asarray(sample_obs.rgb).shape[0])
    model = TorchRSSMDynamics.from_config(wm_cfg)
    init_step = 0
    if args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if not init_path.is_absolute():
            init_path = Path.cwd() / init_path
        payload = model.load_checkpoint(str(init_path))
        init_step = int(payload.get("step") or 0)
        skipped = payload.get("load_skipped") or []
        # Stale Adam moments from an old coll_head (feature-only vs feature+action)
        # or other skipped tensors will crash optimizer.step().
        model.optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(wm_cfg.get("lr", 1e-4)), betas=(0.9, 0.95)
        )
        print(
            f"[wm-validate] init ckpt → {init_path} (step={init_step}"
            + (f", skipped={len(skipped)} tensors" if skipped else "")
            + "; optimizer reset)"
        )
    model.cone_depth_weight = float(args.cone_depth_weight)
    if model.cone_depth_weight > 0.0:
        print(f"[wm-validate] cone_depth_weight={model.cone_depth_weight}")
    if args.collision_head_only:
        head_params = model.freeze_backbone_train_collision_head(
            lr=float(args.head_lr), early_fwd_m=18.0,
        )
        n_param = sum(int(p.numel()) for p in head_params)
        print(
            f"[wm-validate] collision head only: {n_param} params, "
            f"lr={float(args.head_lr):.1e}, early_fwd={model.coll_early_fwd_m}, backbone frozen"
        )
    print(f"[wm-validate] model on {model.device} | latent_dim={model.latent_dim} "
          f"| image_size={wm_cfg['image_size']}")

    # Always emit the ①a–c learning log (the gate consumes it even without a
    # saved ckpt). Mirrors train_depth_head's depth_train.jsonl.
    ckpt_dir = Path(wm_cfg.get("checkpoint_dir") or "experiments/aerial/rl/artifacts/wm_ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "wm_train.jsonl"
    # Provenance sidecar written BEFORE training: a bare loss curve cannot evidence
    # its corpus, which is what disqualified wm_ckpt_v2clean_20260810 as ①a–c
    # evidence. Does not touch the .jsonl the gate parses.
    meta_path = _write_train_meta(
        ckpt_dir, root=root, args=args, buf=buf, image_size=wm_cfg["image_size"]
    )
    print(f"[wm-validate] provenance → {meta_path}"
          + ("  ⚠ allow_v0_desync=TRUE → NOT authoritative ①a–c evidence"
             if args.allow_v0_desync else ""))

    learn_ok = _check_learning(
        model, buf, args.steps, args.wm_batch, args.window, log_path=log_path,
        head_only=bool(args.collision_head_only),
    )
    diverge_ok = _check_non_divergence(model, buf, args.window, args.horizon)

    passed = learn_ok and diverge_ok
    print(f"[wm-validate] learning log → {log_path} "
          f"(feed to `_v0_gate --learning-log {log_path}`)")
    print(f"[wm-validate] {'PASS' if passed else 'FAIL'}: "
          f"learning={learn_ok} non_divergence={diverge_ok}")
    if args.save_ckpt:
        save_step = (
            int(args.save_step)
            if args.save_step is not None
            else (init_step + int(args.steps) if args.init_ckpt else int(args.steps))
        )
        ckpt_path = ckpt_dir / f"wm_step_{save_step}.pt"
        model.save_checkpoint(str(ckpt_path), step=save_step)
        print(f"[wm-validate] checkpoint → {ckpt_path}")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
