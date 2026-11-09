#!/usr/bin/env python3
"""Open an interior Isaac scene and place a Unitree Go2 in it.

AirVLN's packaged pawns are the quadrotor and an SUV, so the dog body cannot
live in that city. This stage is a different scene: Isaac Sim's simple room,
with the Go2 articulation as the body. Run it with Isaac Sim 5.1's Python.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path("/home/yao/dog_airsim/isaac_go2_room.png")
ROOM = "/Isaac/Environments/Simple_Room/simple_room.usd"
GO2 = "/Isaac/Robots/Unitree/Go2/go2.usd"


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 960, "height": 540})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.rotations import gf_quat_to_np_array, lookat_to_quatf
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Usd, UsdGeom

        root = get_assets_root_path()
        print(f"assets_root {root}", flush=True)
        world = World(stage_units_in_meters=1.0)
        add_reference_to_stage(root + ROOM, "/World/Room")
        add_reference_to_stage(root + GO2, "/World/Go2")
        stage = world.stage
        go2 = stage.GetPrimAtPath("/World/Go2")
        if not go2.IsValid():
            print("Go2 prim missing", file=sys.stderr)
            return 2
        xf = UsdGeom.Xformable(go2)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.35))
        names = [str(p.GetPath()) for p in Usd.PrimRange(go2)]
        print(f"go2_prims {len(names)}", flush=True)
        for path in names[:12]:
            print(f"  {path}", flush=True)

        eye = np.array([2.4, -2.2, 1.15], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.35], dtype=np.float32)
        quat = lookat_to_quatf(Gf.Vec3f(*eye.tolist()), Gf.Vec3f(*target.tolist()), Gf.Vec3f(0.0, 0.0, 1.0))
        cam = Camera(
            prim_path="/World/Cam",
            name="cam",
            resolution=(960, 540),
            position=eye,
            orientation=gf_quat_to_np_array(quat),
            frequency=20,
        )
        world.reset()
        cam.initialize()
        for _ in range(8):
            world.step(render=True)
        rgb = np.asarray(cam.get_rgb())
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            print(f"bad frame {getattr(rgb, 'shape', None)}", file=sys.stderr)
            return 3
        frame = np.clip(rgb[..., :3], 0, 255).astype(np.uint8)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            Image.fromarray(frame).save(OUT)
        except Exception:
            OUT.with_suffix(".ppm").write_bytes(
                f"P6\n{frame.shape[1]} {frame.shape[0]}\n255\n".encode() + frame.tobytes()
            )
            print(f"wrote {OUT.with_suffix('.ppm')}", flush=True)
            return 0
        print(f"wrote {OUT} shape={frame.shape}", flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
