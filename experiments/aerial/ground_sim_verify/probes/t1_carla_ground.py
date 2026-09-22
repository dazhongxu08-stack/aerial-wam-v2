#!/usr/bin/env python3
"""T1 — CARLA ground vehicle: RGB, depth, odom, collision, physics, continuous RGB."""
from __future__ import annotations

import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import report  # noqa: E402

HOST = os.environ.get("CARLA_HOST", "127.0.0.1")
PORT = int(os.environ.get("CARLA_PORT", "2200"))
VEHICLE_BP = os.environ.get("GROUND_VEHICLE_BP", "vehicle.tesla.model3")
W = int(os.environ.get("GROUND_RGB_W", "224"))
H = int(os.environ.get("GROUND_RGB_H", "224"))
N = int(os.environ.get("GROUND_PROBE_N", "15"))
DT = float(os.environ.get("GROUND_PROBE_DT_S", "0.05"))
MIN_FPS = float(os.environ.get("GROUND_MIN_FPS", "5.0"))

res: dict = {
    "host": HOST,
    "port": PORT,
    "vehicle_bp": VEHICLE_BP,
    "w": W,
    "h": H,
    "n": N,
    "dt_s": DT,
}


def _probe(name: str, ok: bool, detail: dict, note: str = "") -> None:
    entry = {"pass": bool(ok), "detail": detail}
    if note:
        entry["sanity"] = note
    res[name] = entry


try:
    import carla  # type: ignore
    import numpy as np  # type: ignore
except Exception as e:  # noqa: BLE001
    res["import"] = {"pass": False, "error": repr(e)}
    report.merge("t1_carla_ground", res)
    print("[T1]", res)
    raise SystemExit(1)

actors: list = []
try:
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    veh_bp = bp_lib.find(VEHICLE_BP)
    vehicle = None
    for sp in world.get_map().get_spawn_points():
        vehicle = world.try_spawn_actor(veh_bp, sp)
        if vehicle is not None:
            break
    if vehicle is None:
        raise RuntimeError("vehicle spawn failed")

    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(W))
    rgb_bp.set_attribute("image_size_y", str(H))
    rgb_sensor = world.spawn_actor(
        rgb_bp, carla.Transform(carla.Location(x=1.5, z=1.4)), attach_to=vehicle
    )
    dep_bp = bp_lib.find("sensor.camera.depth")
    dep_bp.set_attribute("image_size_x", str(W))
    dep_bp.set_attribute("image_size_y", str(H))
    dep_sensor = world.spawn_actor(
        dep_bp, carla.Transform(carla.Location(x=1.5, z=1.4)), attach_to=vehicle
    )
    col_bp = bp_lib.find("sensor.other.collision")
    col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
    actors.extend([vehicle, rgb_sensor, dep_sensor, col_sensor])

    rgb_frames: list = []
    dep_frames: list = []
    collided = [False]
    t_rgb: list = []

    def on_rgb(img) -> None:
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((H, W, 4))[:, :, :3]
        rgb_frames.append(arr.copy())
        t_rgb.append(time.perf_counter())

    def on_dep(img) -> None:
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((H, W, 4))
        dep_frames.append(arr.copy())

    def on_col(_evt) -> None:
        collided[0] = True

    rgb_sensor.listen(on_rgb)
    dep_sensor.listen(on_dep)
    col_sensor.listen(on_col)

    world.tick()
    loc0 = vehicle.get_transform().location
    for _ in range(N):
        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
        world.tick()

    loc1 = vehicle.get_transform().location
    vel = vehicle.get_velocity()
    rot = vehicle.get_transform().rotation
    dist = math.hypot(loc1.x - loc0.x, loc1.y - loc0.y)
    speed = math.hypot(vel.x, vel.y)

    rgb_ok = len(rgb_frames) >= 5 and float(np.std(rgb_frames[-1])) > 3.0
    _probe(
        "rgb",
        rgb_ok,
        {"n_frames": len(rgb_frames), "std": float(np.std(rgb_frames[-1])) if rgb_frames else 0},
        f"std={float(np.std(rgb_frames[-1])):.2f}" if rgb_frames else "no frames",
    )

    dep_ok = len(dep_frames) >= 5
    _probe("depth", dep_ok, {"n_frames": len(dep_frames)})

    odom_ok = all(math.isfinite(v) for v in (loc1.x, loc1.y, loc1.z, rot.yaw, vel.x, vel.y))
    _probe(
        "odom",
        odom_ok,
        {
            "loc": (loc1.x, loc1.y, loc1.z),
            "yaw": rot.yaw,
            "vel": (vel.x, vel.y, vel.z),
        },
    )

    _probe("collision", True, {"collided": collided[0], "sensor": "ok"})

    phys_ok = dist > 0.5 or speed > 0.1
    _probe("physics", phys_ok, {"dist_m": dist, "speed_mps": speed})

    fps = 0.0
    mono = True
    if len(t_rgb) >= 2:
        dts = [t_rgb[i] - t_rgb[i - 1] for i in range(1, len(t_rgb))]
        fps = (len(dts) / sum(dts)) if sum(dts) > 0 else 0.0
        mono = all(d > 0 for d in dts)
    cont_ok = fps >= MIN_FPS and mono and rgb_ok
    _probe(
        "continuous_rgb",
        cont_ok,
        {"fps": fps, "monotonic": mono, "min_fps": MIN_FPS},
        f"fps={fps:.2f}",
    )

    res["pass"] = all(res[k]["pass"] for k in ("rgb", "depth", "odom", "collision", "physics", "continuous_rgb"))

finally:
    for a in reversed(actors):
        try:
            a.stop()
            a.destroy()
        except Exception:  # noqa: BLE001
            pass
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    except Exception:  # noqa: BLE001
        pass

report.merge("t1_carla_ground", res)
print("[T1]", {k: res[k].get("pass") if isinstance(res.get(k), dict) else res.get(k) for k in res})
raise SystemExit(0 if res.get("pass") else 1)
