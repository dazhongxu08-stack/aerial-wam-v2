"""Shared flight helpers for Tello calibration / probe scripts (ground console :8090)."""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from experiments.aerial.rl.tello_consequence_policy import altitude_rc
from experiments.aerial.scripts import tello_consequence_collect as gs

RC_HZ = 10.0


def set_base(base: str) -> None:
    gs.BASE = str(base)


def height_m(st: dict) -> Optional[float]:
    h = st.get("height_cm")
    return None if h is None else float(h) / 100.0


def _sleep_rest(t_loop: float) -> None:
    remain = 1.0 / RC_HZ - (time.time() - t_loop)
    if remain > 0:
        time.sleep(remain)


def snapshot_jpeg() -> bytes:
    req = urllib.request.Request(gs.BASE + "/snapshot.jpg", headers={"Connection": "close"})
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        return resp.read()


def settle(band: Tuple[float, float] = (0.85, 1.45), hold_s: float = 2.0, timeout_s: float = 20.0) -> float:
    """Hold a=b=d=0 with altitude trim until height stays in ``band`` for ``hold_s``."""
    t0 = time.time()
    ok_since: Optional[float] = None
    while time.time() - t0 < timeout_s:
        t_loop = time.time()
        st = gs.status()
        gs._gate(st)
        h = height_m(st)
        if h is not None and band[0] <= h <= band[1]:
            ok_since = ok_since or time.time()
            if time.time() - ok_since >= hold_s:
                return float(h)
        else:
            ok_since = None
        gs.rc(0, 0, altitude_rc(h), 0)
        _sleep_rest(t_loop)
    raise RuntimeError(f"settle timeout: height never stayed in {band}")


def hold(
    a: int,
    b: int,
    c: int,
    d: int,
    seconds: float,
    *,
    alt_trim: bool = False,
    stop: Optional[Callable[[dict, float], Optional[str]]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Send a fixed stick for ``seconds``. ``stop(st, t)`` returning a reason ends early."""
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    while True:
        t_loop = time.time()
        t = t_loop - t0
        if t >= seconds:
            return rows, None
        st = gs.status()
        gs._gate(st)
        rows.append({"t": round(t, 3), "yaw": st.get("yaw"), "height_cm": st.get("height_cm")})
        if stop is not None:
            why = stop(st, t)
            if why:
                gs.rc(0, 0, 0, 0)
                return rows, why
        cc = altitude_rc(height_m(st)) if alt_trim else c
        gs.rc(a, b, cc, d)
        _sleep_rest(t_loop)


def imu_hit(run_dir: Optional[Path]) -> str:
    """'' / 'near_miss' / 'collision' from the newest recorded telemetry row."""
    from experiments.aerial.rl.tello_consequence_policy import contact_from_imu

    row = gs.last_telemetry(run_dir) if run_dir is not None else {}
    tel = dict(row.get("tello") or {})
    rc_row = row.get("rc") or [0, 0, 0, 0]

    def f(k: str) -> Optional[float]:
        try:
            return float(tel[k])
        except (KeyError, TypeError, ValueError):
            return None

    h = f("h")
    return contact_from_imu(
        agx=f("agx"),
        agy=f("agy"),
        agz=f("agz"),
        pitch_deg=f("pitch"),
        vgx_dms=f("vgx"),
        rc_b=int(rc_row[1]) if len(rc_row) > 1 else 0,
        height_m=None if h is None else h / 100.0,
    )


def start_record(meta: Dict[str, Any], *, mode: str) -> Tuple[Path, float]:
    gs._req("POST", "/api/control_mode", {"mode": mode})
    rec = gs._req("POST", "/api/record", {"on": True})
    t_rec0 = time.time()
    run_dir = Path(str(rec.get("dir") or ""))
    if not run_dir.is_dir():
        raise RuntimeError(f"record did not start: {rec}")
    gs.patch_meta(run_dir, meta)
    print("record", run_dir, flush=True)
    return run_dir, t_rec0


def finish(run_dir: Optional[Path], events: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """Zero sticks, stop recording, then append events (recorder owns the file while on)."""
    try:
        gs.rc(0, 0, 0, 0)
    except Exception:
        pass
    try:
        print("saved", gs._req("POST", "/api/record", {"on": False}).get("msg"), flush=True)
    except Exception as exc:
        print("record stop failed", exc, flush=True)
    try:
        gs._req("POST", "/api/control_mode", {"mode": "manual"})
    except Exception:
        pass
    if run_dir is not None and run_dir.is_dir():
        gs.append_events(run_dir, events)
        gs.patch_meta(run_dir, meta)


def ask_float(prompt: str) -> Optional[float]:
    try:
        raw = input(prompt).strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("not a number, skipped")
        return None


def read_meta(run_dir: Path) -> Dict[str, Any]:
    p = Path(run_dir) / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
