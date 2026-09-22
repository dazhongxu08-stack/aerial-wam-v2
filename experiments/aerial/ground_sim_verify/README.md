# ground_sim_verify — Ground sim GO/NO-GO (G0 gate)

Analogous to `experiments/aerial/sim_verify/` for **ground / UGV** backends.

**Primary target (84):** Avant-AirSim / CarlaAir — CARLA port `2200`, AirSim port `41463`.

## Quick start (on 84)

```bash
source /data/linux/workspace/miniconda3/etc/profile.d/conda.sh
conda activate carlaAir
cd ~/Projects/aerial-wam-v2/experiments/aerial/ground_sim_verify
cp config.env.example config.env   # edit CARLA_PORT if needed
./run_all.sh
```

Exit code: `0` = Fork G-A (ground GO), `2` = Fork G-A⁻, `3` = Fork G-B.

## Candidates evaluated

| Backend | Repo | Status |
|---------|------|--------|
| **Avant-AirSim (CarlaAir)** | Internal tarball on 84 | **Fork A aerial + CARLA ground sensors — primary** |
| AvantAGSim (GitLab) | `avant/AI-Lab/VLM/AvantAGSim` | Not cloned (GitLab auth); likely same stack as Avant-AirSim |
| slam-nav | `avant/AI-Dev4/dexhand/slam-nav` | Not cloned (GitLab auth); SLAM-centric, secondary |

## Fork G-A criteria

- CARLA connect + vehicle spawn
- RGB 224×224, std > 3, fps ≥ 5
- Depth camera frames (supervision)
- Odometry from vehicle transform + velocity
- Collision sensor readable
- Physics: vehicle moves under throttle (sync mode tick)
