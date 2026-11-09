#!/usr/bin/env python3
"""Short dog walk: one goal a few metres ahead, krs6 scored on measured motion.

Stops on its own at the goal, on a timeout, or when H12 takes the sticks.
With bypass enabled, a blocked front latches the open side until the body has
shifted past it. The floor closer than the lens (about 0.8 m) is not visible.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from aerial_finder.backends.lumos_dog import LumosDogBackend
from aerial_finder.nav.depth import Da3Checkpoint, MetricDa3
from aerial_finder.nav.navigator import ConsequenceNavigator, NavConfig, blocked_ahead


def _build(cfg: dict, base: Path):
    bc = dict(cfg["backend"])
    bc.pop("type", None)
    dc = dict(cfg["depth"])
    backend = LumosDogBackend(**bc)
    try:
        depth_path = dc["path"]
        if not Path(depth_path).is_absolute():
            depth_path = str(base / depth_path)
        depth = Da3Checkpoint(depth_path, device=dc.get("device", "cuda"))
        if dc.get("metric_scale") is not None:
            depth = MetricDa3(depth, float(dc["metric_scale"]), float(dc.get("invalid_m", 30.0)))
        nc = dict(cfg["nav"])
        nc.pop("type", None)
        device = nc.pop("device", "cuda")
        surv = []
        for p in nc.pop("surv_ckpts"):
            q = Path(p)
            surv.append(str(q if q.is_absolute() else base / q))
        nc.pop("prog_ckpts", None)
        nav = ConsequenceNavigator(surv, device=device, cfg=NavConfig(**nc))
    except Exception:
        backend.close()
        raise
    return backend, depth, nav


def hold_if_close(action, clear, near_m):
    """Stand only when ``blocked_ahead`` says the aisle is gone.

    A pallet beside the body during a pass is expected. Stopping on that single
    side cancels a bypass whose front and far side are still open.
    """
    action = np.asarray(action, np.float64)
    if near_m is None or not clear:
        return action, False
    front, left, right = (float(v) for v in clear)
    if blocked_ahead(front, left, right, near_m):
        return np.zeros(4, np.float64), True
    return action, False


class _Trace:
    """Append-only record so a crash still leaves the samples already written."""

    def __init__(self, directory: Path, bridge: str) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "depth").mkdir(exist_ok=True)
        self.steps = (self.dir / "steps.jsonl").open("a", buffering=1)
        self.imu_f = (self.dir / "imu.jsonl").open("a", buffering=1)
        self.health_f = (self.dir / "health.jsonl").open("a", buffering=1)
        self.bridge = bridge
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sensors, daemon=True)
        self._thread.start()

    def _sensors(self) -> None:
        try:
            import sys
            sdk = Path("/home/ubuntu/lumos_sdk/lcm_typedef/python")
            if sdk.is_dir() and str(sdk) not in sys.path:
                sys.path.insert(0, str(sdk))
            import lcm
            from imu_data_lcmt import imu_data_lcmt
            lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
            last = 0.0
            health_at = 0.0

            def on_imu(_ch, data):
                nonlocal last
                now = time.time()
                if now - last < 0.05:
                    return
                last = now
                m = imu_data_lcmt.decode(data)
                self.imu_f.write(json.dumps({
                    "t": now,
                    "rpy": [round(float(v), 5) for v in m.navRPY],
                    "acc": [round(float(v), 4) for v in m.acc],
                    "omega": [round(float(v), 4) for v in m.navOmega],
                }) + "\n")

            lc.subscribe("lcm_imu_data", on_imu)
            while not self._stop.is_set():
                lc.handle_timeout(100)
                now = time.time()
                if now - health_at >= 0.5:
                    health_at = now
                    try:
                        raw = urllib.request.urlopen(self.bridge + "/api/health", timeout=0.3).read()
                        rec = json.loads(raw)
                        rec["t"] = now
                        self.health_f.write(json.dumps(rec) + "\n")
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001
            self.imu_f.write(json.dumps({"t": time.time(), "error": str(exc)}) + "\n")

    def step(self, rec: dict, depth) -> None:
        self.steps.write(json.dumps(rec) + "\n")
        if depth is not None:
            np.save(self.dir / "depth" / f"{int(rec['step']):05d}.npy", np.asarray(depth, np.float16))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        for f in (self.steps, self.imu_f, self.health_f):
            f.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--meters", type=float, default=4.0)
    ap.add_argument("--goal-x", type=float, default=None)
    ap.add_argument("--goal-y", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--reach", type=float, default=1.0)
    ap.add_argument("--record-dir", default="")
    ap.add_argument("--near-m", type=float, default=None)
    ap.add_argument("--near-hold-s", type=float, default=3.0)
    args = ap.parse_args(argv)
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    backend, depth, nav = _build(cfg, cfg_path.parent)
    trace = _Trace(Path(args.record_dir), backend.bridge) if args.record_dir else None
    try:
        backend.prepare()
        depth.reset()
        nav.reset()
        frame, st = backend.observe()
        yaw = float(st.yaw)
        if args.goal_x is None:
            goal = st.pos + np.array([args.meters * math.cos(yaw), args.meters * math.sin(yaw), 0.0])
        else:
            # x forward and y to the integration-frame side, both at the heading when this process starts.
            goal = np.array([float(args.goal_x), float(args.goal_y), 0.0])
        nav.set_band(float(st.pos[2]), float(goal[2]))
        if trace is not None:
            (trace.dir / "meta.json").write_text(json.dumps({
                "meters": args.meters, "timeout_s": args.timeout, "reach_m": args.reach,
                "near_m": args.near_m, "goal_xy": goal[:2].round(3).tolist(),
                "config": str(cfg_path), "t0": time.time(),
            }, indent=1), encoding="utf-8")
        print(f"goal {goal[:2].round(2).tolist()} timeout {args.timeout:.0f}s record {args.record_dir or '-'}", flush=True)
        t0 = time.time()
        last_a = None
        step = 0
        close_since = None
        while time.time() - t0 < args.timeout:
            if step:
                frame, st = backend.observe()
            dist = float(np.linalg.norm(st.pos[:2] - goal[:2]))
            if dist <= args.reach:
                print(f"reached step {step} pos {st.pos[:2].round(2).tolist()} dist {dist:.2f}", flush=True)
                return 0
            if getattr(depth, "uses_valid", False):
                dep, env = depth(frame.rgb, frame.valid)
            else:
                dep, env = depth(frame.rgb)
            done = backend.executed(last_a) if last_a is not None else None
            nav.observe(dep, env, done)
            action = nav.act(st, goal)
            info = getattr(nav, "last", {})
            action, held = hold_if_close(action, info.get("clear_flr"), args.near_m)
            now = time.time()
            if held:
                close_since = now if close_since is None else close_since
            else:
                close_since = None
            print(
                f"step {step} pos {np.round(st.pos[:2], 2).tolist()} dist {dist:.2f} "
                f"a {np.round(action, 3).tolist()} {info.get('family')} surv {info.get('surv')} "
                f"guard {info.get('guard')} clear {info.get('clear_flr')} lat {info.get('bypass_lat')} "
                f"hold {held} pub {backend.last_reply.get('state')} {backend.last_reply.get('vx')} {backend.last_reply.get('vyaw')}",
                flush=True,
            )
            if trace is not None:
                dlast = nav.dhist[-1] if nav.dhist else None
                trace.step({
                    "t": now, "step": step,
                    "pos": [round(float(v), 3) for v in st.pos[:2]],
                    "yaw": round(float(st.yaw), 4), "dist": round(dist, 3),
                    "action": [round(float(v), 4) for v in action],
                    "family": info.get("family"), "guard": info.get("guard"),
                    "surv": None if info.get("surv") is None else round(float(info["surv"]), 4),
                    "clear": info.get("clear_flr"), "lat": info.get("bypass_lat"),
                    "hold": held,
                    "pub_state": backend.last_reply.get("state"),
                    "pub_vx": backend.last_reply.get("vx"),
                    "pub_vyaw": backend.last_reply.get("vyaw"),
                }, dlast)
            backend.step(action)
            if float(max(abs(action[0]), abs(action[3]))) > 1e-3 and not backend.accepted:
                print("walk command not accepted", flush=True)
                return 3
            if held and close_since is not None and now - close_since >= args.near_hold_s:
                print(
                    f"too close step {step} pos {np.round(st.pos[:2], 2).tolist()} clear {info.get('clear_flr')}",
                    flush=True,
                )
                return 4
            last_a = action
            step += 1
        print(f"timeout pos {np.round(st.pos[:2], 2).tolist()}", flush=True)
        return 2
    finally:
        if trace is not None:
            trace.close()
        backend.hover()
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
