#!/usr/bin/env python3
"""Phase-2 + vgoal real-aircraft deploy (Pixhawk 6C or Tello EDU).

Loads the same π / WM / depth / shield stack as ``wam_vgoal_eval``, but drives
``PixhawkDroneEnv`` or ``TelloDroneEnv``. Default is **bench-safe**: mock
camera, no OFFBOARD, no ARM — only loads models and prints one observation.

Stages (Pixhawk):
  (default)       connect MAVLink + load stack + one observe()
  --offboard      enter ArduPilot GUIDED / PX4 OFFBOARD (disarmed)
  --arm           arm motors (requires --i-know-props-are-on)
  --run           closed-loop steps (requires --offboard; --arm for flight)

Tello (Orin on Tello Wi-Fi):

  python -m experiments.aerial.scripts.wam_vgoal_deploy \\
    --backend tello --camera tello --demo-short \\
    --vgoal-repo ~/Projects/aerial-vgoal-wam \\
    --offboard --arm --run --i-know-props-are-on

Recording (Pixhawk default): **ch8** start, **ch9** stop.
Tello defaults to ``--record-auto``. Disable: ``--no-record``.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("wam_vgoal_deploy")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-2 vgoal real-aircraft deploy")
    p.add_argument("--config", default="configs/aerial_rl.yaml")
    p.add_argument("--vgoal-repo", default="~/Projects/aerial-vgoal-wam")
    p.add_argument(
        "--backend",
        choices=("pixhawk", "tello"),
        default="pixhawk",
        help="Flight backend: Pixhawk MAVLink or Tello EDU Wi-Fi SDK",
    )
    p.add_argument("--mavlink-port", default="/dev/ttyACM0")
    p.add_argument("--mavlink-baud", type=int, default=57600)
    p.add_argument("--tello-ip", default="192.168.10.1")
    p.add_argument("--tello-local-ip", default="", help="Override local bind IP for Tello")
    p.add_argument("--step-hz", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument(
        "--camera",
        default="0",
        help="V4L2 device, 'tello' stream, or use --mock-camera",
    )
    p.add_argument("--mock-camera", action="store_true")
    p.add_argument("--capture-w", type=int, default=1280)
    p.add_argument("--capture-h", type=int, default=720)
    p.add_argument("--capture-fps", type=int, default=30)
    p.add_argument("--wam-encode-size", type=int, default=224)
    p.add_argument("--camera-fov-deg", type=float, default=80.0)
    p.add_argument("--cruise-speed", type=float, default=10.0)
    p.add_argument("--tti-coeff", type=float, default=2.5)
    p.add_argument("--success-dist", type=float, default=3.0)
    p.add_argument("--yolo-model", default="yolov8n.pt")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-imgsz", type=int, default=1280)
    p.add_argument("--target-class", default="car")
    p.add_argument("--visual-prompt", default=None)
    p.add_argument("--goal-x", type=float, default=None)
    p.add_argument("--goal-y", type=float, default=None)
    p.add_argument("--goal-z", type=float, default=None)
    p.add_argument("--fallback-toward-g", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--offboard", action="store_true")
    p.add_argument("--arm", action="store_true")
    p.add_argument("--run", action="store_true", help="Execute closed-loop steps")
    p.add_argument("--i-know-props-are-on", action="store_true")
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
        "--tau-ckpt",
        default="experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--no-depth-shield",
        action="store_true",
        help="Disable TTI depth shield (recommended for 24F bench / untrusted depth)",
    )
    p.add_argument(
        "--record-dir",
        default=None,
        help="Orin local save root (default: <repo>/artifacts/orin_deploy)",
    )
    p.add_argument("--no-record", action="store_true", help="Disable frame + state recording")
    p.add_argument(
        "--record-auto",
        action="store_true",
        help="Start recording immediately (default: wait for RC ch8/ch9)",
    )
    p.add_argument("--record-start-ch", type=int, default=8, help="RC channel to start recording")
    p.add_argument("--record-stop-ch", type=int, default=9, help="RC channel to stop recording")
    p.add_argument("--corpus-scene", default="real_hardware", help="Corpus scene tag")
    p.add_argument("--corpus-handover-id", default=None, help="Optional handover_id for corpus")
    p.add_argument("--corpus-leg", default="deploy", help="Corpus leg tag (outdoor/indoor/deploy)")
    p.add_argument("--corpus-instruction", default=None, help="Free-text corpus instruction")
    p.add_argument(
        "--demo-short",
        action="store_true",
        help="Short visual approach preset (detect target, fly ~15-25m, no GPS fallback)",
    )
    p.add_argument(
        "--preset",
        choices=("none", "v5"),
        default="none",
        help="v5 = hard-route sealed baseline (hbclear + open_loop face/peel planner + shield)",
    )
    p.add_argument("--planner", action="store_true", help="Enable ImaginationPlanner")
    p.add_argument("--planner-horizon", type=int, default=5)
    p.add_argument(
        "--planner-rollout",
        choices=("open_loop", "closed_loop"),
        default="open_loop",
        help="open_loop = v5 hand-rule face/peel; closed_loop = actor tail, no hand bias",
    )
    p.add_argument(
        "--shield-exclusion-forward-only",
        action="store_true",
        help="Three-zone shield uses forward cone only (v5 gate default)",
    )
    p.add_argument(
        "--success-on-visual",
        action="store_true",
        help="Success when visual target within --success-dist (not annot goal)",
    )
    p.add_argument(
        "--detect-bench",
        action="store_true",
        help="Bench only: camera + YOLO loop, no WM/flight (props off)",
    )
    p.add_argument("--detect-bench-steps", type=int, default=40)
    p.add_argument(
        "--search-det-steer",
        action="store_true",
        help="When SEARCHING, steer toward YOLO bbox before area search",
    )
    p.add_argument(
        "--reject-far-lock-m",
        type=float,
        default=0.0,
        help="Reject visual locks farther than this (0=off)",
    )
    return p.parse_args()


def _apply_preset_v5(args: argparse.Namespace) -> None:
    """Hard-route sealed baseline (2026-09-21 hbclear_facegoal_v5, SR 3/4)."""
    args.actor_ckpt = (
        "experiments/aerial/rl/artifacts/"
        "v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear/v4_ac_latest.pt"
    )
    args.wm_ckpt = (
        "experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt"
    )
    args.depth_ckpt = (
        "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/"
        "depth_best_holdout_da3_ft_head.pt"
    )
    args.tau_ckpt = (
        "experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt"
    )
    args.planner = True
    args.planner_horizon = int(args.planner_horizon or 5)
    args.planner_rollout = "open_loop"
    args.no_depth_shield = False
    args.shield_exclusion_forward_only = True
    args.tti_coeff = 2.5
    if str(getattr(args, "backend", "pixhawk")) != "tello":
        args.cruise_speed = 10.0
    if args.corpus_instruction is None:
        args.corpus_instruction = "v5 hard-route baseline → vgoal"
    logger.info(
        "preset=v5 actor=hbclear wm=depth_aux_long planner=open_loop H=%d shield=on",
        int(args.planner_horizon),
    )


def _apply_demo_short(args: argparse.Namespace) -> None:
    """Preset for short outdoor visual approach demo."""
    args.fallback_toward_g = False
    args.success_on_visual = True
    # Keep shield if preset=v5 (sealed baseline had shield on).
    if str(getattr(args, "preset", "none")) != "v5":
        args.no_depth_shield = True
    args.search_det_steer = True
    args.reject_far_lock_m = 40.0
    args.cruise_speed = 4.0
    args.max_steps = 120
    args.success_dist = 4.0
    args.step_hz = 5.0
    args.yolo_conf = max(0.2, float(args.yolo_conf))
    if args.corpus_instruction is None:
        args.corpus_instruction = f"short demo: approach {args.target_class}"
    if args.corpus_scene == "real_hardware":
        args.corpus_scene = "real_outdoor_demo"
    if str(getattr(args, "backend", "pixhawk")) == "tello":
        args.cruise_speed = 0.5
        args.success_dist = 1.5
        args.reject_far_lock_m = 8.0
        args.max_steps = 80
        args.capture_w = 960
        args.capture_h = 720
        if args.camera == "0":
            args.camera = "tello"
        if args.corpus_scene == "real_outdoor_demo":
            args.corpus_scene = "tello_indoor_demo"
        if args.corpus_instruction and "short demo" in str(args.corpus_instruction):
            args.corpus_instruction = f"tello short demo: approach {args.target_class}"
        if str(getattr(args, "preset", "none")) == "v5":
            args.corpus_instruction = f"v5→tello: approach {args.target_class}"
        if not args.record_auto and not args.no_record:
            args.record_auto = True


def _visual_success(
    args: argparse.Namespace,
    vstep: Any,
    p_curr: np.ndarray,
) -> tuple[bool, str]:
    from experiments.aerial.scripts.wam_phase2_long_eval import _goal_dist
    from vgoal.tracker import TargetState

    if not args.success_on_visual or vstep.target_world is None:
        return False, ""
    d_vis = float(_goal_dist(p_curr, vstep.target_world))
    if d_vis <= float(args.success_dist):
        return True, f"visual target dist={d_vis:.1f}m"
    if vstep.tracker_state == TargetState.ARRIVED.value:
        return True, "tracker ARRIVED"
    if vstep.goal_rel is not None and len(np.asarray(vstep.goal_rel).reshape(-1)) >= 4:
        d_rel = float(np.asarray(vstep.goal_rel).reshape(-1)[3])
        if d_rel <= float(args.success_dist):
            return True, f"goal_rel dist={d_rel:.1f}m"
    return False, ""


def _run_detect_bench(args: argparse.Namespace, root: Path, vgoal_repo: Path) -> int:
    from experiments.aerial.deploy.real_camera import RealCamera, RealCameraConfig
    from experiments.aerial.scripts.wam_vgoal_eval import _build_detector

    logger.info(
        "detect-bench: target=%s steps=%d (place object in FOV)",
        args.target_class,
        int(args.detect_bench_steps),
    )
    cam = RealCamera(
        RealCameraConfig(
            device=str(args.camera),
            width=int(args.capture_w),
            height=int(args.capture_h),
            fps=int(args.capture_fps),
            wam_size=int(args.wam_encode_size),
        )
    )
    cam.open()
    detector = _build_detector(
        argparse.Namespace(
            detector="yolo",
            target_class=args.target_class,
            visual_prompt=args.visual_prompt or args.target_class,
            yolo_model=args.yolo_model,
            yolo_conf=args.yolo_conf,
            yolo_imgsz=int(args.yolo_imgsz),
            yolo_device=str(args.device),
            camera_fov_deg=args.camera_fov_deg,
            capture_w=args.capture_w,
            capture_h=args.capture_h,
        ),
        vgoal_repo,
    )
    import cv2  # type: ignore

    hits = 0
    try:
        for step in range(int(args.detect_bench_steps)):
            bgr, _rgb = cam.read()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detect_all = getattr(detector, "detect_all", None)
            if callable(detect_all):
                dets = list(detect_all(rgb) or [])
            else:
                det = detector.detect(rgb)
                dets = list(det) if isinstance(det, (list, tuple)) else ([det] if det is not None else [])
            if dets:
                hits += 1
                det0 = dets[0]
                bb = getattr(det0, "bbox", None)
                conf = float(getattr(det0, "confidence", 0.0) or 0.0)
                logger.info("step=%02d HIT n=%d conf=%.2f bbox=%s", step, len(dets), conf, bb)
            elif step % 5 == 0:
                logger.info("step=%02d no %s in view", step, args.target_class)
            time.sleep(1.0 / max(1.0, float(args.step_hz)))
    finally:
        cam.close()
    logger.info("detect-bench done: %d/%d frames with %s", hits, args.detect_bench_steps, args.target_class)
    return 0 if hits > 0 else 1


def _goal_from_args(args: argparse.Namespace, origin: np.ndarray) -> Optional[np.ndarray]:
    if args.goal_x is None or args.goal_y is None or args.goal_z is None:
        return None
    return np.array([args.goal_x, args.goal_y, args.goal_z], dtype=np.float64)


def _corpus_meta(args: argparse.Namespace, annot_goal: Optional[np.ndarray]) -> dict[str, Any]:
    return {
        "source": "orin_deploy",
        "map_id": "real_world",
        "scene": str(args.corpus_scene),
        "leg": str(args.corpus_leg),
        "handover_id": args.corpus_handover_id,
        "goal_world": annot_goal.tolist() if annot_goal is not None else None,
        "target_class": str(args.target_class),
        "instruction": args.corpus_instruction,
        "camera": str(args.camera),
        "mock_camera": bool(args.mock_camera),
    }


def _poll_record_gate(
    gate: Any,
    bridge: Any,
    recorder: Optional[Any],
    *,
    record_base: Path,
    manifest: dict[str, Any],
) -> tuple[Any | None, str]:
    """Process one RC sample for ch8/ch9 record control. Returns (recorder, event)."""
    from experiments.aerial.deploy.orin_deploy_recorder import OrinDeployRecorder

    bridge.poll(timeout_s=0.0)
    event = gate.update_rc_dict(bridge.rc_pwm_dict())
    if event == "start":
        if recorder is not None:
            recorder.close()
        recorder = OrinDeployRecorder.open_run(record_base, manifest)
        logger.info("RC ch%d: recording START → %s", gate.config.start_channel, recorder.root)
    elif event == "stop" and recorder is not None:
        path = recorder.root
        recorder.close()
        logger.info("RC ch%d: recording STOP → %s", gate.config.stop_channel, path)
        recorder = None
    return recorder, event


def _deploy_manifest(args: argparse.Namespace, annot_goal: Optional[np.ndarray]) -> dict[str, Any]:
    return {
        "script": "wam_vgoal_deploy",
        "backend": str(args.backend),
        "mavlink_port": args.mavlink_port,
        "tello_ip": getattr(args, "tello_ip", None),
        "step_hz": float(args.step_hz),
        "max_steps": int(args.max_steps),
        "offboard": bool(args.offboard),
        "arm": bool(args.arm),
        "run": bool(args.run),
        "corpus": _corpus_meta(args, annot_goal),
        "checkpoints": {
            "wm": str(args.wm_ckpt),
            "actor": str(args.actor_ckpt),
            "depth": str(args.depth_ckpt),
            "tau": str(args.tau_ckpt),
        },
    }


def main() -> int:
    args = _parse()
    if str(args.preset) == "v5":
        _apply_preset_v5(args)
    if args.demo_short:
        _apply_demo_short(args)
    if args.arm and not args.i_know_props_are_on:
        logger.error("Refusing --arm without --i-know-props-are-on")
        return 2
    if args.run and not args.offboard:
        logger.error("--run requires --offboard")
        return 2
    if args.arm and not args.offboard:
        logger.error("--arm requires --offboard")
        return 2

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    vgoal_repo = Path(args.vgoal_repo).expanduser().resolve()
    if not vgoal_repo.is_dir():
        raise SystemExit(f"--vgoal-repo not found: {vgoal_repo}")
    if str(vgoal_repo) not in sys.path:
        sys.path.insert(0, str(vgoal_repo))

    if args.detect_bench:
        device_str = str(args.device)
        import torch

        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        args.device = device_str
        return _run_detect_bench(args, root, vgoal_repo)

    from vgoal.geometry import CameraIntrinsics
    from vgoal.tracker import TargetTracker, TrackerConfig

    import torch
    from experiments.aerial.rl.actor_critic import LatentActorCritic, LatentActorDeployPolicy
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.action import body_delta_limits, clip_body_delta
    from experiments.aerial.rl.planner import ImaginationPlanner
    from experiments.aerial.rl.reward import RewardConfig
    from experiments.aerial.rl.scene_intent import TowardGoalIntent
    from experiments.aerial.rl.tau_predictor import make_tau_predictor
    from experiments.aerial.rl.train_rl import _build_safety, load_torch_dynamics
    from experiments.aerial.deploy.orin_deploy_recorder import OrinDeployRecorder
    from experiments.aerial.rl.env.orin_rc_handoff import RcRecordGate, RcRecordGateConfig
    from experiments.aerial.scripts.wam_vgoal_eval import _build_detector, _vision_step

    cfg = yaml.safe_load((root / args.config).read_text()) if (root / args.config).is_file() else {}
    device_str = str(args.device)
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    if device_str == "cuda":
        try:
            torch.zeros(1, device="cuda")
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA unusable (%s) — falling back to cpu", exc)
            device_str = "cpu"
    device = torch.device(device_str)
    logger.info(
        "deploy backend=%s device=%s mock_camera=%s",
        args.backend,
        device,
        args.mock_camera,
    )

    if args.backend == "tello":
        from experiments.aerial.rl.env.tello_env import TelloDroneEnv, TelloEnvConfig

        env = TelloDroneEnv(
            TelloEnvConfig(
                tello_ip=str(args.tello_ip),
                local_ip=str(args.tello_local_ip or ""),
                step_hz=float(args.step_hz),
                camera_device=str(args.camera),
                capture_w=int(args.capture_w),
                capture_h=int(args.capture_h),
                capture_fps=int(args.capture_fps),
                wam_encode_size=int(args.wam_encode_size),
                takeoff_on_reset=bool(args.arm),
                control_on_reset=bool(args.offboard),
                mock_camera=bool(args.mock_camera),
                max_v_mps=float(min(1.0, max(0.2, float(args.cruise_speed)))),
            )
        )
    else:
        from experiments.aerial.rl.env.pixhawk_env import PixhawkDroneEnv, PixhawkEnvConfig

        env = PixhawkDroneEnv(
            PixhawkEnvConfig(
                mavlink_port=args.mavlink_port,
                mavlink_baud=int(args.mavlink_baud),
                step_hz=float(args.step_hz),
                camera_device=str(args.camera),
                capture_w=int(args.capture_w),
                capture_h=int(args.capture_h),
                capture_fps=int(args.capture_fps),
                wam_encode_size=int(args.wam_encode_size),
                offboard_on_reset=bool(args.offboard),
                arm_on_reset=bool(args.arm),
                mock_camera=bool(args.mock_camera),
            )
        )

    wm_cfg = cfg.get("world_model") or {}
    wm_path = (root / args.wm_ckpt).resolve()
    dynamics, _ = load_torch_dynamics(wm_cfg, str(wm_path), device=device_str, success_dist_m=float(args.success_dist))

    actor_path = (root / args.actor_ckpt).resolve()
    actor_ac = LatentActorCritic.load_from_checkpoint(actor_path, device=device_str)
    actor_ac.config.goal_feat_mode = "meter"
    policy = LatentActorDeployPolicy(dynamics, actor_ac, deterministic=True, stream_latent=True)

    depth_path = (root / args.depth_ckpt).resolve()
    depth_pred = DepthMinPredictor.from_checkpoint(depth_path, device=device_str)
    tau_pred = make_tau_predictor(
        kind="foe_calibrated",
        ckpt=(root / args.tau_ckpt).resolve(),
        device=device_str,
    )

    dt_step = 1.0 / float(args.step_hz)
    phys_limits = body_delta_limits(dt_step)
    vx_max_step = float(min(float(args.cruise_speed) / float(args.step_hz), float(phys_limits[0])))
    action_limits = np.array(
        [vx_max_step, float(phys_limits[1]), float(phys_limits[2]), float(phys_limits[3])],
        dtype=np.float64,
    )

    reward_cfg = RewardConfig(**(cfg.get("reward") or {}))
    reward_cfg.success_dist_m = float(args.success_dist)

    planner = None
    if bool(args.planner):
        planner = ImaginationPlanner(
            dynamics=dynamics,
            horizon=int(args.planner_horizon),
            reward_cfg=reward_cfg,
            action_limits=action_limits,
            rollout_mode=str(args.planner_rollout),
            tail_policy=policy if str(args.planner_rollout) == "closed_loop" else None,
        )
        logger.info(
            "planner rollout=%s horizon=%d hand_bias=%s",
            args.planner_rollout,
            int(args.planner_horizon),
            "off" if str(args.planner_rollout) == "closed_loop" else "on(v5)",
        )

    safety_cfg = dict(cfg.get("safety") or {})
    if args.no_depth_shield:
        safety_cfg["kind"] = "null"
    elif str(safety_cfg.get("kind", "null")) in ("null", "none", "None"):
        safety_cfg["kind"] = "three_zone"
    if bool(getattr(args, "shield_exclusion_forward_only", False)):
        safety_cfg["exclusion_forward_only"] = True
        logger.info("shield exclusion: forward cone only")
    safety_cfg["v_cruise_m_s"] = float(args.cruise_speed)
    safety_cfg["tti_coeff"] = float(args.tti_coeff)
    shield = _build_safety(safety_cfg)

    fallback_intent = (
        TowardGoalIntent(r_m=100.0, mode="toward_g", cruise_speed=float(args.cruise_speed))
        if args.fallback_toward_g
        else None
    )

    intrinsics = CameraIntrinsics.from_fov(
        float(args.camera_fov_deg),
        width=int(args.capture_w),
        height=int(args.capture_h),
    )
    tracker = TargetTracker(
        TrackerConfig(
            success_dist_m=float(args.success_dist),
            max_occlusion_s=2.0,
            min_confidence=float(max(0.15, args.yolo_conf)),
        )
    )
    detector = _build_detector(
        argparse.Namespace(
            detector="yolo",
            target_class=args.target_class,
            visual_prompt=args.visual_prompt or args.target_class,
            yolo_model=args.yolo_model,
            yolo_conf=args.yolo_conf,
            yolo_imgsz=int(args.yolo_imgsz),
            yolo_device=device_str,
            camera_fov_deg=args.camera_fov_deg,
            capture_w=args.capture_w,
            capture_h=args.capture_h,
        ),
        vgoal_repo,
    )

    recorder: OrinDeployRecorder | None = None
    record_base = (
        Path(args.record_dir).expanduser()
        if args.record_dir
        else (root / "artifacts" / "orin_deploy")
    )
    record_enabled = not args.no_record
    record_gate: RcRecordGate | None = None
    deploy_manifest: dict[str, Any] | None = None
    if record_enabled:
        record_gate = RcRecordGate(
            RcRecordGateConfig(
                start_channel=int(args.record_start_ch),
                stop_channel=int(args.record_stop_ch),
            )
        )
    try:
        obs = env.reset()
        p_curr = np.asarray(obs.position, dtype=np.float64)
        curr_yaw = float(obs.yaw)
        annot_goal = _goal_from_args(args, p_curr)
        if annot_goal is None:
            if args.fallback_toward_g:
                annot_goal = p_curr + np.array([20.0, 0.0, 0.0], dtype=np.float64)
                logger.warning("No --goal-x/y/z; using fallback annot_goal=%s", annot_goal.tolist())
            else:
                annot_goal = p_curr.copy()
                logger.info("Visual-only mode: no annot goal (target=%s)", args.target_class)

        deploy_manifest = _deploy_manifest(args, annot_goal)
        if record_enabled and args.record_auto:
            recorder = OrinDeployRecorder.open_run(record_base, deploy_manifest)
            record_gate.active = True
            logger.info("Auto-recording to %s", recorder.root)
        elif record_enabled:
            logger.info(
                "Recording armed — ch%d=start, ch%d=stop (independent of ch7 Orin handoff)",
                int(args.record_start_ch),
                int(args.record_stop_ch),
            )

        policy.reset()
        shield.reset()
        tau_pred.reset()
        depth_pred.reset()
        if planner is not None:
            planner.reset()
        if fallback_intent is not None:
            fallback_intent.reset()

        logger.info(
            "obs pos=%s yaw=%.1f° armed=%s offboard=%s",
            [round(float(x), 2) for x in p_curr],
            math.degrees(curr_yaw),
            env._bridge.is_armed(),
            env._offboard_active,
        )

        if not args.run:
            logger.info("Bench load OK — pass --run --offboard to close the loop")
            if (
                record_enabled
                and record_gate is not None
                and deploy_manifest is not None
                and args.backend == "pixhawk"
            ):
                logger.info("Waiting for RC record buttons (Ctrl+C to exit)")
                try:
                    while True:
                        recorder, event = _poll_record_gate(
                            record_gate,
                            env._bridge,
                            recorder,
                            record_base=record_base,
                            manifest=deploy_manifest,
                        )
                        if recorder is not None and record_gate.active:
                            obs = env.observe()
                            obs.info["corpus"] = _corpus_meta(args, annot_goal)
                            recorder.record_step(
                                obs,
                                step_info={"phase": "bench", "rc_event": event},
                            )
                        time.sleep(1.0 / max(1.0, float(args.step_hz)))
                except KeyboardInterrupt:
                    logger.warning("Bench record loop interrupted")
            return 0

        search_fwd_step = min(0.5 / float(args.step_hz), vx_max_step)
        p_prev = p_curr.copy()
        prev_yaw = curr_yaw

        for step in range(int(args.max_steps)):
            if (
                record_enabled
                and record_gate is not None
                and deploy_manifest is not None
                and args.backend == "pixhawk"
            ):
                recorder, _ = _poll_record_gate(
                    record_gate,
                    env._bridge,
                    recorder,
                    record_base=record_base,
                    manifest=deploy_manifest,
                )

            obs.info.pop("depth_min_pred", None)
            obs.info.pop("tau_pred", None)
            d_fwd = depth_pred.predict_min(obs)
            if d_fwd is not None:
                obs.info["depth_min_pred"] = float(d_fwd)
            tau_v = tau_pred.predict_tau(obs)
            if tau_v is not None:
                obs.info["tau_pred"] = float(tau_v)

            vstep = _vision_step(
                obs=obs,
                detector=detector,
                tracker=tracker,
                dynamic_tracker=None,
                dynamic_min_meas_conf=float(max(0.15, args.yolo_conf)),
                depth_pred=depth_pred,
                intrinsics=intrinsics,
                pos=p_curr,
                yaw=curr_yaw,
                prev_pos=p_prev,
                prev_yaw=prev_yaw,
                dt=dt_step,
                search_fwd_step=search_fwd_step,
                search_yaw_rate=0.1,
                area_search_planner=None,
                fallback_intent=fallback_intent,
                annot_goal=annot_goal,
                allow_fallback=bool(args.fallback_toward_g),
                prefer_nearest=False,
                camera_fov_deg=float(args.camera_fov_deg),
                reject_far_lock_m=float(args.reject_far_lock_m),
                search_det_steer=bool(args.search_det_steer),
            )
            p_prev = p_curr.copy()
            prev_yaw = curr_yaw

            if vstep.search_action is not None:
                action = vstep.search_action
                goal_rel = None
            elif vstep.goal_rel is not None:
                goal_rel = np.asarray(vstep.goal_rel, dtype=np.float32)
                z = dynamics.encode(obs)
                action = policy.act_latent(z, goal_rel)
                if planner is not None:
                    # Keep obs.info goal_rel for face/peel hand-rule scoring (v5).
                    obs.info["goal_rel"] = goal_rel.tolist()
                    if vstep.target_world is not None:
                        planner.set_goal(np.asarray(vstep.target_world, dtype=np.float64))
                        obs.info["goal"] = np.asarray(
                            vstep.target_world, dtype=np.float64
                        ).tolist()
                    action = planner.plan(obs, action, latent=z)
                    action = clip_body_delta(action, action_limits)
            else:
                action = np.zeros(4, dtype=np.float64)
                goal_rel = None

            wm_out = None
            if depth_pred is not None:
                wm_out = {"depth_min_pred": obs.info.get("depth_min_pred")}
            act_safe, overridden = shield.apply_action(action, obs, wm_out=wm_out, limits=action_limits)
            if overridden:
                action = act_safe

            obs, step_info = env.step(action)
            p_curr = np.asarray(obs.position, dtype=np.float64)
            curr_yaw = float(obs.yaw)

            if recorder is not None and record_gate is not None and record_gate.active:
                obs.info["corpus"] = _corpus_meta(args, annot_goal)
                gr = goal_rel.tolist() if goal_rel is not None else None
                recorder.record_step(
                    obs,
                    step_info={
                        "phase": "control",
                        "action": [float(x) for x in np.asarray(action).reshape(-1)],
                        "action_overridden": bool(overridden),
                        "goal_rel": gr,
                        "tracker_state": vstep.tracker_state,
                        "using_fallback": bool(vstep.using_fallback),
                        "depth_min_pred": obs.info.get("depth_min_pred"),
                        "tau_pred": obs.info.get("tau_pred"),
                        **step_info,
                    },
                )

            if step % max(1, int(args.step_hz)) == 0:
                gr = goal_rel.tolist() if goal_rel is not None else None
                logger.info(
                    "step=%04d pos=%s tracker=%s fallback=%s goal_rel=%s ir=%s",
                    step,
                    [round(float(x), 1) for x in p_curr],
                    vstep.tracker_state,
                    vstep.using_fallback,
                    [round(float(x), 2) for x in gr] if gr else None,
                    step_info.get("armed"),
                )

            ok, why = _visual_success(args, vstep, p_curr)
            if ok:
                logger.info("SUCCESS %s at step %d", why, step)
                break
            if args.fallback_toward_g and annot_goal is not None:
                dist = float(np.linalg.norm(annot_goal - p_curr))
                if dist <= float(args.success_dist):
                    logger.info("SUCCESS annot dist=%.1fm at step %d", dist, step)
                    break

        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    finally:
        if recorder is not None:
            recorder.close()
            logger.info("Recording closed: %s", recorder.root)
        env.close()


if __name__ == "__main__":
    sys.exit(main())
