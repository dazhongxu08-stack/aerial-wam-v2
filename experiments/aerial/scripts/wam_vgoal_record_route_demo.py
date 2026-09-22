#!/usr/bin/env python3
"""Waypoint-following demo video with YOLO overlay (option B).

Flies the annotation polyline via AdaptiveSubgoal + Phase-2 π + planner + shield
(same as ``wam_phase2_record_route``). Records 1080p ego video with YOLO boxes.

Not a pure visual closed loop — the path is anchored to the route; YOLO is overlay only.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from experiments.aerial.scripts.wam_phase2_record_route import (
    _goal_dist,
    _write_frames_ffmpeg,
    _write_traj_plot,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("wam_vgoal_record_route_demo")


def _draw_yolo_overlay(
    bgr: np.ndarray,
    det: Any,
    *,
    target_class: str,
) -> np.ndarray:
    if det is None:
        return bgr
    bb = np.asarray(det.bbox, dtype=np.float64).reshape(-1)
    if bb.size < 4:
        return bgr
    x1, y1, x2, y2 = [int(round(float(v))) for v in bb[:4]]
    color = (0, 220, 80)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    conf = float(getattr(det, "confidence", 0.0) or 0.0)
    cv2.putText(
        bgr,
        f"{target_class} {conf:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return bgr


def _draw_demo_hud(
    bgr: np.ndarray,
    *,
    step: int,
    max_steps: int,
    target_class: str,
    pos: np.ndarray,
    d_goal: float,
    rem_dist: float,
    prog: float,
    cte: float,
    det_hit: bool,
    shield: bool,
) -> np.ndarray:
    img = bgr.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.putText(
        img,
        f"VGoal waypoint demo | YOLO {target_class} | step {step}/{max_steps}",
        (12, 22),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    ctrl = "SHIELD" if shield else "policy"
    det_s = "DET" if det_hit else "no-det"
    cv2.putText(
        img,
        f"prog={prog*100:.1f}% rem={rem_dist:.1f}m d_goal={d_goal:.1f}m CTE={cte:.1f}m {det_s} {ctrl}",
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return img


def main() -> int:
    p = argparse.ArgumentParser(description="Waypoint route demo + YOLO overlay video")
    p.add_argument("--route-idx", type=int, default=0)
    p.add_argument("--config", default="configs/aerial_rl.yaml")
    p.add_argument(
        "--wm-ckpt",
        default="experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt",
    )
    p.add_argument(
        "--actor-ckpt",
        default=(
            "experiments/aerial/rl/artifacts/"
            "v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt"
        ),
    )
    p.add_argument(
        "--depth-ckpt",
        default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt",
    )
    p.add_argument(
        "--annotation",
        default="experiments/aerial/annotations/sim_vgoal_demo_short_route.json",
    )
    p.add_argument("--cruise-speed", type=float, default=3.0, help="Urban crowded default: 3 m/s")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--step-hz", type=float, default=5.0)
    p.add_argument("--success-dist", type=float, default=4.0)
    p.add_argument("--planner-horizon", type=int, default=5)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--out-dir", default="artifacts/sim_vgoal_demo_short")
    p.add_argument("--out-mp4", default=None, help="Default: <out-dir>/demo_waypoint_yolo.mp4")
    p.add_argument("--vgoal-repo", default=os.path.expanduser("~/aerial-vgoal-wam"))
    p.add_argument("--target-class", default="car")
    p.add_argument("--yolo-model", default="yolov8n.pt")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--yolo-device", default="cuda")
    p.add_argument("--capture-w", type=int, default=1920)
    p.add_argument("--capture-h", type=int, default=1080)
    p.add_argument("--wam-encode-size", type=int, default=224)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    vgoal_repo = Path(args.vgoal_repo).expanduser().resolve()
    if not vgoal_repo.is_dir():
        raise SystemExit(f"--vgoal-repo not found: {vgoal_repo}")
    if str(vgoal_repo) not in sys.path:
        sys.path.insert(0, str(vgoal_repo))

    from experiments.aerial.scripts.wam_vgoal_eval import _build_detector
    import torch
    from experiments.aerial.rl.actor_critic import LatentActorCritic, LatentActorDeployPolicy
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.action import body_delta_limits, clip_body_delta
    from experiments.aerial.rl.goal_features import body_vel_from_obs
    from experiments.aerial.rl.planner import ImaginationPlanner
    from experiments.aerial.rl.reward import RewardConfig
    from experiments.aerial.rl.subgoal_generator import AdaptiveSubgoalGenerator
    from experiments.aerial.rl.train_rl import _build_env, _build_safety, load_torch_dynamics

    cfg = yaml.safe_load((root / args.config).read_text())
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    anno_path = Path(args.annotation)
    if not anno_path.is_absolute():
        anno_path = root / anno_path
    with open(anno_path, "r", encoding="utf-8") as f:
        anno = json.load(f)
    routes = anno.get("routes", anno) if isinstance(anno, dict) else anno
    r_info = routes[int(args.route_idx)]
    pts = np.array(r_info.get("pos", r_info.get("positions")), dtype=np.float64)
    yaws = np.array(r_info.get("yaw", [0.0] * len(pts)), dtype=np.float64)
    goal_pos = pts[-1].copy()
    start_yaw = float(yaws[0]) if len(yaws) else 0.0
    ref_len = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

    env_cfg = dict(cfg.get("env") or {})
    env_cfg["backend"] = "airsim"
    env_cfg["step_hz"] = float(args.step_hz)
    env_cfg["grab_depth"] = True
    env_cfg["fanout_rgb"] = True
    env_cfg["width"] = int(args.capture_w)
    env_cfg["height"] = int(args.capture_h)
    env_cfg["wam_encode_size"] = int(args.wam_encode_size)
    env = _build_env(env_cfg)

    dynamics, _ = load_torch_dynamics(
        cfg.get("world_model") or {},
        str(root / args.wm_ckpt),
        device=device_str,
        success_dist_m=float(args.success_dist),
    )
    actor_ac = LatentActorCritic.load_from_checkpoint(str(root / args.actor_ckpt), device=device_str)
    actor_ac.config.goal_feat_mode = "meter"

    phys = body_delta_limits(1.0 / float(args.step_hz))
    action_limits = np.array(
        [
            min(float(args.cruise_speed) / float(args.step_hz), float(phys[0])),
            float(phys[1]),
            float(phys[2]),
            float(phys[3]),
        ],
        dtype=np.float64,
    )
    reward_cfg = RewardConfig(**(cfg.get("reward") or {}))
    reward_cfg.success_dist_m = float(args.success_dist)
    planner = ImaginationPlanner(
        dynamics=dynamics,
        horizon=int(args.planner_horizon),
        reward_cfg=reward_cfg,
        action_limits=action_limits,
    )
    policy = LatentActorDeployPolicy(dynamics, actor_ac, deterministic=True, stream_latent=True)

    depth_path = root / args.depth_ckpt
    depth_pred = (
        DepthMinPredictor.from_checkpoint(str(depth_path), device=device_str)
        if depth_path.is_file()
        else None
    )
    safety_cfg = dict(cfg.get("safety") or {})
    if str(safety_cfg.get("kind", "null")) in ("null", "none", "None"):
        safety_cfg["kind"] = "three_zone"
    if "three_zone" not in safety_cfg:
        safety_cfg["three_zone"] = {}
    safety_cfg["three_zone"]["v_cruise"] = float(args.cruise_speed)
    shield = _build_safety(safety_cfg)
    subgoal_gen = AdaptiveSubgoalGenerator(cruise_speed=float(args.cruise_speed))
    subgoal_gen.reset()
    if hasattr(shield, "reset"):
        shield.reset()
    if depth_pred is not None:
        depth_pred.reset()
    policy.reset()

    # _build_detector expects wam_vgoal_eval argparse fields.
    args.detector = "yolo"
    args.visual_prompt = None
    detector = _build_detector(args, vgoal_repo)

    ep_dict: Dict[str, Any] = {
        "pos": pts.tolist(),
        "yaw": yaws.tolist() if len(yaws) == len(pts) else [start_yaw] * len(pts),
        "gpt_instruction": r_info.get("gpt_instruction", "waypoint demo"),
    }
    obs = env.reset(ep_dict)
    p_curr = np.array(obs.position, dtype=np.float64)
    curr_yaw = float(obs.yaw) if hasattr(obs, "yaw") else start_yaw
    logger.info(
        "Waypoint+YOLO demo route %02d L_ref=%.1fm start_d=%.1fm capture=%dx%d",
        args.route_idx + 1,
        ref_len,
        _goal_dist(p_curr, goal_pos),
        int(args.capture_w),
        int(args.capture_h),
    )

    frames: List[np.ndarray] = []
    traj: List[np.ndarray] = [p_curr.copy()]
    min_d = _goal_dist(p_curr, goal_pos)
    arrived = False
    interventions = 0
    det_hits = 0
    s_prog = 0.0
    rem_dist = ref_len
    cte = 0.0
    stride = max(1, int(args.frame_stride))

    for step in range(int(args.max_steps)):
        d_fwd = None
        if depth_pred is not None and obs.rgb is not None:
            pred_both = getattr(depth_pred, "predict_min_and_cones", None)
            if callable(pred_both):
                d_min_pred, cones = pred_both(obs)
                if d_min_pred is not None:
                    obs.info["depth_min_pred"] = float(d_min_pred)
                if isinstance(cones, dict):
                    obs.info["depth_cones_pred"] = {
                        k: (float(v) if v is not None else None) for k, v in cones.items()
                    }
                    cf = cones.get("forward")
                    if cf is not None and np.isfinite(float(cf)):
                        d_fwd = float(cf)

        g_rel_body, s_info = subgoal_gen.compute_subgoal(
            curr_pos=p_curr,
            curr_yaw=curr_yaw,
            global_path=pts,
            d_fwd_hat=d_fwd,
        )
        target_world = np.array(s_info["target_world"], dtype=np.float64)
        s_prog = float(s_info["s_progress"])
        rem_dist = float(s_info["rem_dist"])
        cte = float(s_info.get("cte_m", 0.0))
        safe_v = float(s_info.get("safe_speed_limit", args.cruise_speed))

        phys = body_delta_limits(1.0 / float(args.step_hz))
        cur_limits = np.array(
            [
                float(min(safe_v / float(args.step_hz), float(phys[0]))),
                float(phys[1]),
                float(phys[2]),
                float(phys[3]),
            ],
            dtype=np.float64,
        )
        planner.action_limits = cur_limits
        d_to_goal = _goal_dist(p_curr, goal_pos)
        min_d = min(min_d, d_to_goal)
        prog = float(np.clip(s_prog / max(1e-3, ref_len), 0.0, 1.0))

        det = None
        det_hit = False
        rgb_det = getattr(obs, "rgb_yolo", None)
        if rgb_det is None:
            rgb_det = obs.rgb
        if rgb_det is not None:
            det = detector.detect(np.asarray(rgb_det, dtype=np.uint8))
            det_hit = det is not None

        if not args.no_video and rgb_det is not None and step % stride == 0:
            bgr = cv2.cvtColor(np.asarray(rgb_det, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            bgr = _draw_yolo_overlay(bgr, det, target_class=str(args.target_class))
            bgr = _draw_demo_hud(
                bgr,
                step=step,
                max_steps=int(args.max_steps),
                target_class=str(args.target_class),
                pos=p_curr,
                d_goal=d_to_goal,
                rem_dist=rem_dist,
                prog=prog,
                cte=cte,
                det_hit=det_hit,
                shield=False,
            )
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if det_hit:
            det_hits += 1

        if rem_dist <= float(args.success_dist) and d_to_goal <= float(args.success_dist):
            arrived = True
            break

        obs.info["goal"] = target_world.tolist()
        obs.info["goal_rel"] = g_rel_body.tolist()
        planner.set_goal(target_world)
        action = policy.act(obs)
        action = planner.plan(obs, action, latent=policy._latent)
        action = clip_body_delta(action, cur_limits)

        wm_out = None
        if policy._latent is not None and hasattr(dynamics, "step"):
            try:
                wm_out = dynamics.step(
                    policy._latent,
                    action,
                    goal_rel=g_rel_body,
                    body_vel=body_vel_from_obs(obs),
                )
            except Exception:
                wm_out = None

        overridden = False
        if shield is not None:
            action, overridden = shield.apply_action(action, obs, wm_out=wm_out, limits=cur_limits)
            if overridden:
                interventions += 1

        step_out = env.step(action)
        if len(step_out) == 4:
            obs, _rew, done, step_info = step_out
        else:
            obs, step_info = step_out
            done = bool(getattr(obs, "collided", False))

        p_curr = np.array(obs.position, dtype=np.float64)
        curr_yaw = float(obs.yaw) if hasattr(obs, "yaw") else curr_yaw
        traj.append(p_curr.copy())
        if done:
            break

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    flown = np.asarray(traj, dtype=np.float64)
    tag = f"route{args.route_idx:02d}"
    traj_json = out_dir / f"{tag}_waypoint_traj.json"
    traj_png = out_dir / f"{tag}_waypoint_traj_xy.png"
    traj_json.write_text(
        json.dumps(
            {"route_idx": int(args.route_idx), "ref_polyline": pts.tolist(), "flown": flown.tolist()},
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_traj_plot(
        pts,
        flown,
        traj_png,
        title=f"Waypoint demo R{args.route_idx + 1:02d}  prog={prog*100:.1f}%  d_min={min_d:.1f}m",
    )

    out_mp4 = Path(args.out_mp4) if args.out_mp4 else out_dir / "demo_waypoint_yolo.mp4"
    if not args.no_video:
        if not frames:
            raise RuntimeError("no RGB frames captured")
        _write_frames_ffmpeg(frames, out_mp4 if out_mp4.is_absolute() else root / out_mp4, fps=float(args.fps))

    summary = {
        "mode": "waypoint_b_yolo_overlay",
        "route_idx": int(args.route_idx),
        "ref_len_m": round(ref_len, 2),
        "steps": int(len(flown)),
        "d_min_m": round(min_d, 2),
        "d_final_m": round(_goal_dist(p_curr, goal_pos), 2),
        "progress_ratio": round(prog, 4),
        "arrived": bool(arrived),
        "detection_frac": round(det_hits / max(1, len(flown)), 4),
        "intervention_rate": round(interventions / max(1, len(flown)), 4),
        "video_path": str(out_mp4),
        "traj_json": str(traj_json),
        "traj_plot": str(traj_png),
        "stack": "AdaptiveSubgoal + toward_g π + planner + ThreeZone + YOLO overlay",
        "actor_ckpt": str(args.actor_ckpt),
    }
    summary_path = out_dir / f"{tag}_waypoint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("SUMMARY %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
