"""Offline conversion of tello_gs JPEG+jsonl → schema-v2 npz."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.aerial.rl.convert_tello_gs import convert_runs, run_to_transitions
from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.tello_qr import render_qr_bgr


def _write_jpeg(path: Path, val: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 48), color=(int(val), 40, 80))
    img.putpixel((0, 0), ((int(val) + 80) % 256, 200, 10))
    img.save(path, format="JPEG", quality=90)


def _fake_run(root: Path, *, emergency: bool = False, airborne: bool = True) -> Path:
    run = root / "run_test"
    frames = run / "frames"
    frames.mkdir(parents=True)
    tel_lines = []
    for i in range(6):
        seq = i + 1
        _write_jpeg(frames / f"{seq:06d}.jpg", 20 + i * 30)
        tel_lines.append(
            {
                "t_s": 0.2 * i,
                "frame_seq": seq,
                "airborne": airborne,
                "control_mode": "auto",
                "rc": [0, 22, 0, 0] if airborne else [0, 0, 0, 0],
                "pose": {"x_m": 0.1 * i, "y_m": 0.0, "z_m": 0.9, "yaw_deg": 0.0},
                "tello": {
                    "yaw": 0,
                    "vgx": 3,
                    "vgy": 0,
                    "vgz": 0,
                    "h": 90,
                    "agx": -2.0,
                    "agy": 0.0,
                    "agz": -1000.0,
                },
            }
        )
    (run / "telemetry.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in tel_lines), encoding="utf-8"
    )
    events = [{"t_s": 0.0, "event": "record_start"}]
    if emergency:
        events.append({"t_s": 1.0, "event": "emergency"})
    events.append({"t_s": 1.2, "event": "record_stop"})
    (run / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    return run


def test_run_to_transitions_rgb224_no_depth(tmp_path: Path) -> None:
    run = _fake_run(tmp_path)
    trans = run_to_transitions(run, hz=0.0)
    assert len(trans) >= 3
    arr = ds.episode_arrays(trans)
    assert arr["rgb"].shape[1:] == (224, 224, 3)
    assert "depth" not in arr
    assert arr["actions"].shape[1] == 4
    assert float(arr["actions"][0, 0]) > 0.0  # forward stick b=22
    assert bool(arr["imu_present"].any())
    assert not bool(arr["collided"].any())


def test_emergency_marks_last_collided(tmp_path: Path) -> None:
    run = _fake_run(tmp_path, emergency=True)
    trans = run_to_transitions(run, hz=0.0)
    arr = ds.episode_arrays(trans)
    assert bool(arr["collided"][-1])
    assert bool(trans[-1].done)


def test_grounded_run_skipped(tmp_path: Path) -> None:
    run = _fake_run(tmp_path, airborne=False)
    assert run_to_transitions(run, hz=0.0) == []


def test_single_qr_stamps_goal_and_arrived(tmp_path: Path) -> None:
    import cv2  # type: ignore

    run = _fake_run(tmp_path)
    qr = render_qr_bgr("only-one-qr")
    cv2.imwrite(str(run / "frames" / "000006.jpg"), qr)
    trans = run_to_transitions(run, hz=0.0)
    arr = ds.episode_arrays(trans)
    assert "goal" in arr
    np.testing.assert_allclose(arr["goal"], trans[-1].obs.position.astype(np.float32), atol=0.2)
    assert bool(trans[0].info.get("arrived"))
    assert trans[0].info["qr"]["data"] == "only-one-qr"


def test_convert_writes_npz_manifest(tmp_path: Path) -> None:
    src = tmp_path / "tello_gs"
    _fake_run(src)
    out = tmp_path / "npz"
    result = convert_runs(src, out, hz=0.0)
    assert len(result["episodes"]) == 1
    path = out / result["episodes"][0]["file"]
    loaded = np.load(path)
    assert "rgb" in loaded.files and "proprio" in loaded.files
    assert (out / "manifest.json").is_file()
    assert (out / "QUALITY_SUMMARY.json").is_file()
