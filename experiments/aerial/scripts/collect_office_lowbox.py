#!/usr/bin/env python3
"""Place short crates in the office aisle and save a new training pair file.

The crates are spare barrels, scaled on Z so the top stays below the camera.
They write depth. The six aisle barrels already in the gap routes are not moved.
Original transforms go to lowbox_restore.json. Nothing is written into
indoor_office_gap or indoor_office_train.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/simworld/run")
import collect_office_train as col  # noqa: E402

OUT = Path("/data/simworld/run/indoor_office_lowbox")
SCALE_Z = 0.7
# Lane center is y=-200. These sit to the side, clear of the six gap barrels.
BOXES = [
    ("SM_BarelPlastic_A_21", -280.0, -90.0),
    ("SM_BarelPlastic_A_22", -40.0, -90.0),
    ("SM_BarelPlastic_A_24", 180.0, -90.0),
    ("SM_BarelPlastic_A_25", 360.0, -90.0),
    ("SM_BarelPlastic_A_26", -280.0, -310.0),
    ("SM_BarelPlastic_A_27", -40.0, -310.0),
    ("SM_BarelPlastic_A_28", 180.0, -310.0),
    ("SM_BarelPlastic_A_29", 360.0, -310.0),
]


def req(ucv, command: str) -> str:
    return str(col.req(ucv, command))


def save_restore(ucv) -> None:
    rows = []
    for name, _x, _y in BOXES:
        rows.append(
            {
                "name": name,
                "location": req(ucv, f"vget /object/{name}/location"),
                "rotation": req(ucv, f"vget /object/{name}/rotation"),
                "scale": req(ucv, f"vget /object/{name}/scale"),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "lowbox_restore.json"
    if path.exists():
        print("RESTORE_KEPT", flush=True)
        return
    path.write_text(json.dumps(rows, indent=2))
    print(f"RESTORE_SAVED {len(rows)}", flush=True)


def place_boxes(ucv) -> None:
    for name, x, y in BOXES:
        req(ucv, f"vset /object/{name}/object_mobility true")
        req(ucv, f"vset /object/{name}/physics false")
        req(ucv, f"vset /object/{name}/scale 1 1 {SCALE_Z}")
        req(ucv, f"vset /object/{name}/rotation 0 0 0")
        req(ucv, f"vset /object/{name}/location {x} {y} -48")
        time.sleep(0.05)
        print(name, req(ucv, f"vget /object/{name}/bounds"), flush=True)


def cone_of(depth56: np.ndarray) -> tuple[float, float, float]:
    fwd, left, right = [float(v) for v in col.cones3(depth56)]
    return fwd, left, right


def grab_frame(ucv, cam: int, x: float, y: float, yaw: float):
    col.look(ucv, cam, x, y, yaw)
    time.sleep(0.06)
    packed = col.grab(ucv, cam)
    if packed is None:
        return None
    rgb, depth = packed
    if float(rgb.mean()) < 22 or float(rgb.max()) < 40:
        return None
    image, seen, warped = col.warp_pair(rgb, depth)
    d56 = col.pool_min(col.hide(warped, seen))
    if image.shape[:2] != (224, 224):
        return None
    return image, d56, cone_of(d56)


def collect(ucv, cam: int) -> None:
    images, depths, splits, yaws, meta = [], [], [], [], []
    dists = (160.0, 200.0, 240.0, 290.0, 350.0, 420.0)
    cam_ys = (-200.0, -210.0, -190.0)
    n = 0
    for _name, bx, by in BOXES:
        # +Y is image right. Aim a little toward the box so the face stays in the side cone.
        toward = 16.0 if by > -200.0 else -16.0
        yaws_deg = (0.0, toward, toward * 0.5, toward * 1.4)
        for dist in dists:
            for cy in cam_ys:
                for yaw in yaws_deg:
                    n += 1
                    frame = grab_frame(ucv, cam, bx - dist, cy, yaw)
                    if frame is None:
                        continue
                    image, d56, (fwd, left, right) = frame
                    side = min(left, right)
                    other = max(left, right)
                    # One side is the short face. The forward cone still looks down the lane.
                    if not (side < 1.6 and fwd > 1.8 and other > 1.6):
                        continue
                    images.append(image)
                    depths.append(d56)
                    yaws.append(math.radians(yaw))
                    meta.append((bx, by, dist, cy, yaw, fwd, left, right))
    if len(images) < 24:
        print(f"TOO_FEW {len(images)} checked {n}", flush=True)
        return
    hold = np.zeros(len(images), dtype="<U8")
    hold[::7] = "hold"
    hold[hold == ""] = "train"
    path = OUT / "pairs.npz"
    np.savez(
        path,
        rgb=np.stack(images).astype(np.uint8),
        depth56=np.stack(depths).astype(np.float16),
        split=hold,
        yaw=np.asarray(yaws, np.float32),
    )
    (OUT / "frames.json").write_text(
        json.dumps(
            [
                {
                    "bx": a, "by": b, "dist": d, "cy": c, "yaw": y,
                    "fwd": f, "left": l, "right": r,
                }
                for a, b, d, c, y, f, l, r in meta
            ]
        )
    )
    near_side = sum(1 for *_, f, l, r in meta if min(l, r) < 1.3 and f > 2.0)
    print(
        f"SAVED {path} frames {len(images)} hold {int((hold == 'hold').sum())} "
        f"side_near_fwd_open {near_side} checked {n}",
        flush=True,
    )


def main() -> None:
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    if "--restore" in sys.argv:
        rows = json.loads((OUT / "lowbox_restore.json").read_text())
        for row in rows:
            name = row["name"]
            req(ucv, f"vset /object/{name}/object_mobility true")
            req(ucv, f"vset /object/{name}/physics false")
            req(ucv, f"vset /object/{name}/scale {row['scale']}")
            req(ucv, f"vset /object/{name}/rotation {row['rotation']}")
            req(ucv, f"vset /object/{name}/location {row['location']}")
        print("RESTORED", len(rows), flush=True)
        return
    save_restore(ucv)
    place_boxes(ucv)
    collect(ucv, cam=1)


if __name__ == "__main__":
    main()
