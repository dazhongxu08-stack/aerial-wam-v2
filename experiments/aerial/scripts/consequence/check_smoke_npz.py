import json
import cv2
import numpy as np

z = np.load("experiments/aerial/rl/artifacts/consequence_smoke_20261004/episodes/episode_00001.npz")
for k in z.files:
    print(k, z[k].shape, z[k].dtype)
off = z["jpeg_offsets"]
img = cv2.imdecode(z["jpeg"][off[0]:off[1]], cv2.IMREAD_COLOR)
print("jpeg0", img.shape)
meta = json.loads(bytes(z["meta"]).decode())
print("meta", {k: meta[k] for k in ("route_id", "yaw0_off_deg", "steps", "end_reason")})
print("macros", np.unique(z["macro"], return_counts=True))
print("collided tail", z["collided"][-3:], "fwd cone tail", z["cones"][-3:, 0])
print("act[0:3]", z["action"][:3])
