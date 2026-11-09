import time
import numpy as np
import yaml
from experiments.aerial.rl.train_rl import _build_env

cfg = yaml.safe_load(open("configs/_tmp_feasible_side_h125.yaml")) or {}
env_cfg = dict(cfg.get("env") or {})
env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
env = _build_env(env_cfg)
c = env._connect()
print("camera", env.config.camera, "fanout", env.config.fanout_rgb)
parts = {
    "scene": lambda: env._grab_scene_native(c),
    "depth": lambda: env._grab_depth(c),
    "imu": lambda: env._grab_imu(c),
    "collision": lambda: env._grab_collision(c),
    "state": lambda: env.observe_state(),
}
for k, f in parts.items():
    ts = []
    for _ in range(8):
        t = time.time(); out = f(); ts.append(time.time() - t)
    shape = getattr(out, "shape", None)
    print(k, "mean %.3f" % np.mean(ts), shape)
