# Ground Sim Verdict (84) — 20260920_002402

## Selected backend: **Avant-AirSim (CarlaAir)**

- CARLA: `127.0.0.1:2200` (ground vehicle + sensors)
- AirSim: `127.0.0.1:41463` (aerial Fork A, shared world)
- Product: `/data/linux/workspace/Avant-AirSim/Avant-AirSim`

## Candidates

| Repo | Clone | Verdict |
|------|-------|---------|
| Avant-AirSim (deployed) | On 84 | **GO — primary G0 backend** |
| AvantAGSim (GitLab) | Auth blocked | Same product line; use deployed tarball |
| slam-nav (GitLab) | Auth blocked | Deferred; SLAM-centric, not WAM-native |

## Probe exits
- ground_sim_verify: 1
- ground smoke: 0

## Next (G0 execution)
1. `GroundRobotBridge` = `CarlaGroundRobotEnv` (scaffolded)
2. G1: load aerial `step_e` π + WM on 10–50 m ground routes
3. G2: ground RGB + odom + depth GT corpus for DA3 fine-tune
