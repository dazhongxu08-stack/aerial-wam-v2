"""Hold forward speed until an in-place spin's steering has come back near center.

The body spins at 0.40 rad/s with the four wheels near ±44°. A straight
command in the next frame drives those wheels before they return to 0°.
"""
from __future__ import annotations

CENTER_DEG = 8.0
CENTER_WAIT_S = 2.0
SPIN_POSE_DEG = 30.0
SPIN_WZ = 0.25


def hold_for_center(dx: float, prev_vx: float, prev_wz: float, angles, now: float, center_until: float, gave_up: bool = False) -> bool:
    """True when this forward step must wait. A spin that is still spinning does not."""
    if dx <= 0.0 or not angles or gave_up:
        return False
    peak = max(abs(float(a)) for a in angles)
    if peak <= CENTER_DEG:
        return False
    was_spin = float(prev_vx) < 0.05 and abs(float(prev_wz)) >= SPIN_WZ
    # A timed-out wait must not start over on the next frame.
    if float(center_until) > 0.0 and float(now) >= float(center_until) and not was_spin:
        return False
    return was_spin or float(center_until) > float(now) or peak >= SPIN_POSE_DEG


def next_center_state(holding: bool, peak: float, now: float, center_until: float, gave_up: bool):
    """Keep a timed-out wait latched until the wheels are actually near center."""
    if holding:
        if float(center_until) <= float(now):
            center_until = float(now) + CENTER_WAIT_S
        return center_until, gave_up
    if peak <= CENTER_DEG:
        return 0.0, False
    if (float(center_until) > 0.0 and float(now) >= float(center_until)) or gave_up:
        return 0.0, True
    return 0.0, False


def wheel_angles(msg):
    """Four steering angles in degrees, or None when this chassis message has none."""
    try:
        return (
            float(msg.front_angle_fb.front_angle_fb_l),
            float(msg.front_angle_fb.front_angle_fb_r),
            float(msg.rear_angle_fb.rear_angle_fb_l),
            float(msg.rear_angle_fb.rear_angle_fb_r),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def install(module) -> None:
    """Patch a loaded wire so a forward command waits out a crooked spin."""
    import json
    import time

    import numpy as np

    wire = module.YhsWire
    orig_chassis = wire._on_chassis
    orig_publish = wire.publish_action
    gear = int(getattr(module, "SPEED_GEAR", 6))

    def _on_chassis(self, msg) -> None:
        orig_chassis(self, msg)
        angles = wheel_angles(msg)
        if angles is None:
            return
        with self.lock:
            self.steer_deg = angles

    def publish_action(self, action: np.ndarray, vx_cap=None) -> None:
        from geometry_msgs.msg import Twist

        raw = np.asarray(action, np.float64)
        now = time.time()
        with self.lock:
            prev = self.last_twist
            angles = getattr(self, "steer_deg", None)
            until = float(getattr(self, "_center_until", 0.0))
            gave_up = bool(getattr(self, "_center_gave_up", False))
        prev_vx = 0.0 if prev is None else float(prev.linear.x)
        prev_wz = 0.0 if prev is None else float(prev.angular.z)
        if float(raw[0]) <= 0.0:
            gave_up = False
        with self.lock:
            joypad = bool(getattr(self, "joypad", False))
        if joypad and angles and max(abs(float(a)) for a in angles) >= SPIN_POSE_DEG and float(raw[0]) > 0.0:
            gave_up = False
            until = 0.0
        holding = hold_for_center(float(raw[0]), prev_vx, prev_wz, angles, now, until, gave_up)
        peak = 0.0 if not angles else max(abs(float(a)) for a in angles)
        until, gave_up = next_center_state(holding, peak, now, until, gave_up)
        with self.lock:
            self._center_until = until
            self._center_gave_up = gave_up
        if holding:
            twist = Twist()
            cmd = json.dumps(
                {"vx": 0.0, "wz_deg_s": 0.0, "gear": gear, "steer": "center"},
                ensure_ascii=False,
            )
            with self.lock:
                self.last_action = [0.0, 0.0, 0.0, 0.0]
                self.last_twist = twist
                self.last_cmd = cmd
                self.model_step = np.zeros(4, np.float32)
                self.yaw_tgt = None
            if self.armed:
                self.pub.publish(twist)
            return
        orig_publish(self, action, vx_cap)

    wire._on_chassis = _on_chassis
    wire.publish_action = publish_action
