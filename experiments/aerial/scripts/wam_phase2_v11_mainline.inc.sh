# V11 mainline stack (source from eval scripts; do not execute directly).
# Route-10 peel winner: polyline + rolling-global + heading-assist, planner H=1.
WAM_PHASE2_V11_STACK=(
  --subgoal-source polyline
  --rolling-global
  --heading-assist
  --global-horizon-m 60
  --global-replan-period-s 1.0
  --heading-assist-cte-max-m 8.0
  --heading-assist-cos-thr 0.7
  --planner --planner-horizon 1
  --tti-coeff 2.5
  --heading-reentry-cos 0.5
  --cte-reentry-m 1.5
  --cruise-speed 10.0
  --max-steps 600
)

WAM_PHASE2_CKPTS=(
  --annotation "${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_long_only.json}"
  --wm-ckpt "${WM:-experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt}"
  --actor-ckpt "${ACTOR:-experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt}"
  --depth-ckpt "${DEPTH:-experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
  --tau-ckpt "${TAU:-experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
  --goal-feat-mode meter
)
