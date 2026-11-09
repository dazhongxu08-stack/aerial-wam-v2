import json, time
import numpy as np
import yaml
from experiments.aerial.rl.train_rl import _build_env
from experiments.aerial.rl.collect_consequence_dataset import EpisodeBuffer
from experiments.aerial.rl.spawn_utils import reset_with_spawn_retries

cfg = yaml.safe_load(open("configs/_tmp_feasible_side_h125.yaml")) or {}
env_cfg = dict(cfg.get("env") or {})
env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
print({k: env_cfg.get(k) for k in ("width", "height", "step_hz", "rate_lock", "camera_names", "grab_rgb", "grab_depth", "sim_clock_speed")})
env = _build_env(env_cfg)
r = json.load(open("experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json"))["episodes"][1]
t = time.time()
used, obs, _, _ = reset_with_spawn_retries(env, r, min_spawn_z=24.0, spawn_z_retry_m=5.0, spawn_z_max_retries=4, min_spawn_clear_m=5.0)
print("reset", round(time.time() - t, 2), "depth", None if obs.depth is None else obs.depth.shape, "rgb", obs.rgb.shape)
buf = EpisodeBuffer()
ts, tb = [], []
for i in range(20):
    t = time.time()
    obs, info = env.step(np.array([0.0, 0.0, 0.0, 0.1]))
    ts.append(time.time() - t)
    t = time.time()
    buf.add(obs, np.zeros(4), 0, 0)
    tb.append(time.time() - t)
print("step mean %.3f p90 %.3f  buf mean %.3f" % (np.mean(ts), np.percentile(ts, 90), np.mean(tb)))
print("info keys", sorted(info.keys())[:30])
