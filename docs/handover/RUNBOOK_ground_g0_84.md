# RUNBOOK — Ground G0 on 84 (Avant-AirSim)

## Verdict (2026-09-19)

**Use Avant-AirSim (CarlaAir)** on 84 as the ground WAM simulation backend.

| Check | Avant-AirSim 84 | slam-nav | AvantAGSim GitLab |
|-------|-----------------|----------|-------------------|
| On 84 / runnable | Yes (ports 2200/41463) | Not cloned | Not cloned |
| Aerial Fork A | PASS (IMU/depth/physics) | — | — |
| CARLA ground RGB+depth | PASS (~57 fps RGB) | — | — |
| Ground vehicle examples | `drive_vehicle.py`, `air_ground_sync.py` | — | — |
| WAM bridge fit | High (`CarlaGroundRobotEnv`) | Low (SLAM stack) | TBD (= likely same as deployed) |

## Ports (84)

- CARLA: `127.0.0.1:2200` (public `10.229.20.84:2200`)
- AirSim: `127.0.0.1:41463` (public `10.229.20.84:41463`)
- Map: `HumenCorridor` (default)

## Start sim (if down)

```bash
bash /data/linux/workspace/Avant-AirSim/start_humen.sh
```

## Run G0 probes

```bash
source /data/linux/workspace/miniconda3/etc/profile.d/conda.sh
conda activate carlaAir
cd ~/Projects/aerial-wam-v2/experiments/aerial/ground_sim_verify
./run_all.sh
```

## Overnight pipeline

```bash
cd ~/Projects/aerial-wam-v2
nohup bash experiments/aerial/scripts/run_ground_g0_84_overnight.sh \
  >> logs/ground_g0_84_overnight.log 2>&1 &
tail -f logs/ground_g0_84_overnight.log
```

## G0 code

- `experiments/aerial/rl/env/ground_robot_env.py` — `GroundRobotBridge` / `CarlaGroundRobotEnv`
- `experiments/aerial/ground_sim_verify/` — Fork G-A probes
- Ground action limits: `GROUND_MAX_BODY_VELOCITY` in `env/action.py`
