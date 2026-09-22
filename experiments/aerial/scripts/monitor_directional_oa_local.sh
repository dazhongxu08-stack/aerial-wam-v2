#!/usr/bin/env bash
# Mac-side poller: check 125+84 every INTERVAL; restart host watchdogs if missing;
# emit AGENT_LOOP_TICK when action needed or summary.
set -euo pipefail
INTERVAL="${INTERVAL:-180}"
STAMP125="${STAMP125:-20260922_172751}"
STAMP84="${STAMP84:-20260922_175214}"
while true; do
  sleep "$INTERVAL"
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  out125=$(ssh -o ConnectTimeout=12 -o BatchMode=yes cursor-125-lan "bash -s" <<EOF 2>&1 || echo SSH125_FAIL
set +e
ART=/home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts
STAMP=$STAMP125
DONE=\$(grep -c 'CL-FT-125 DONE' \$ART/logs/directional_oa_cl_ft_\${STAMP}.log 2>/dev/null || true); DONE=\${DONE:-0}
ALIVE=\$(pgrep -af "cl_ft_\${STAMP}|train_v4_ac.*\${STAMP}|long_eval.*\${STAMP}|resume_directional" | grep -vc pgrep || true)
WATCH=0; [[ -f \$ART/logs/watch_cl_ft_125_\${STAMP}.pid ]] && kill -0 \$(cat \$ART/logs/watch_cl_ft_125_\${STAMP}.pid) 2>/dev/null && WATCH=1; [[ \$WATCH -eq 0 ]] && WATCH=\$(pgrep -fc watch_directional_oa_job.sh || true)
# ensure watchdog
if [[ "\$DONE" -eq 0 && "\$WATCH" -eq 0 ]]; then
  cd /home/yao/aerial-wam-v2
  JOB=cl_ft_125 STAMP=\$STAMP INTERVAL=90 nohup bash experiments/aerial/scripts/watch_directional_oa_job.sh \
    >>\$ART/logs/watch_cl_ft_125_\${STAMP}_outer.log 2>&1 &
  echo RESTARTED_WATCH
fi
# if dead and not done, resume now
if [[ "\$DONE" -eq 0 && "\$ALIVE" -eq 0 ]]; then
  cd /home/yao/aerial-wam-v2
  JOB=cl_ft_125 STAMP=\$STAMP nohup bash experiments/aerial/scripts/resume_directional_oa_job.sh \
    >>\$ART/logs/watch_cl_ft_125_\${STAMP}_resume_nohup.log 2>&1 &
  echo RESUMED
fi
# SR snapshot
srs=""
for t in cl_noshield_b0 cl_noshield_b1 cl_noshield_b2 cl_shield_b0 cl_shield_b1 cl_shield_b2 actor_noshield_a0 actor_noshield_a1 actor_noshield_a2; do
  p=\$ART/eval_directional_oa_\${STAMP}_\$t/eval_all.json
  if [[ -f \$p ]]; then
    s=\$(python3 -c "import json;d=json.load(open('\$p'));print(sum(1 for r in d['episodes'] if r.get('arrived')))" 2>/dev/null || echo '?')
    srs="\$srs \$t=\$s/4"
  else
    srs="\$srs \$t=pending"
  fi
done
echo "125 done=\$DONE alive=\$ALIVE watch=\$WATCH\$srs"
EOF
)
  out84=$(ssh -o ConnectTimeout=12 -o BatchMode=yes cursor-84-lan "bash -s" <<EOF 2>&1 || echo SSH84_FAIL
set +e
ART=/data/aerial-wam-v2/experiments/aerial/rl/artifacts
STAMP=$STAMP84
DONE=\$(grep -c 'CL-FT-125 DONE' \$ART/logs/directional_oa_cl_ft_\${STAMP}.log 2>/dev/null || true); DONE=\${DONE:-0}
ALIVE=\$(pgrep -af "cl_ft_\${STAMP}|train_v4_ac.*\${STAMP}|long_eval.*\${STAMP}|resume_directional" | grep -vc pgrep || true)
WATCH=0; [[ -f \$ART/logs/watch_cl_ft_84_\${STAMP}.pid ]] && kill -0 \$(cat \$ART/logs/watch_cl_ft_84_\${STAMP}.pid) 2>/dev/null && WATCH=1; [[ \$WATCH -eq 0 ]] && WATCH=\$(pgrep -fc watch_directional_oa_job.sh || true)
if [[ "\$DONE" -eq 0 && "\$WATCH" -eq 0 ]]; then
  cd /data/aerial-wam-v2
  JOB=cl_ft_125 STAMP=\$STAMP INTERVAL=90 nohup bash experiments/aerial/scripts/watch_directional_oa_job.sh \
    >>\$ART/logs/watch_cl_ft_84_\${STAMP}_outer.log 2>&1 &
  echo RESTARTED_WATCH
fi
if [[ "\$DONE" -eq 0 && "\$ALIVE" -eq 0 ]]; then
  cd /data/aerial-wam-v2
  JOB=cl_ft_125 STAMP=\$STAMP WM=/data/aerial-wam-v2/experiments/aerial/rl/artifacts/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt \
    nohup bash experiments/aerial/scripts/resume_directional_oa_job.sh \
    >>\$ART/logs/watch_cl_ft_84_\${STAMP}_resume_nohup.log 2>&1 &
  echo RESUMED
fi
srs=""
for t in cl_noshield_b0 cl_noshield_b1 cl_noshield_b2 cl_shield_b0 cl_shield_b1 cl_shield_b2 actor_noshield_a0 actor_noshield_a1 actor_noshield_a2; do
  p=\$ART/eval_directional_oa_\${STAMP}_\$t/eval_all.json
  if [[ -f \$p ]]; then
    s=\$(python3 -c "import json;d=json.load(open('\$p'));print(sum(1 for r in d['episodes'] if r.get('arrived')))" 2>/dev/null || echo '?')
    srs="\$srs \$t=\$s/4"
  else
    srs="\$srs \$t=pending"
  fi
done
echo "84 done=\$DONE alive=\$ALIVE watch=\$WATCH\$srs"
EOF
)
  action=0
  echo "$out125" | grep -qE 'RESTARTED_WATCH|RESUMED|SSH125_FAIL' && action=1
  echo "$out84" | grep -qE 'RESTARTED_WATCH|RESUMED|SSH84_FAIL' && action=1
  # Always tick so agent sees progress; flag action in payload
  echo "AGENT_LOOP_TICK_dir_oa {\"prompt\":\"Check 125+84 CL FT B-G; if interrupted resume; report SR/climb. ts=$ts action=$action\",\"125\":$(echo "$out125" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[-800:]))' 2>/dev/null || echo '\"\"'),\"84\":$(echo "$out84" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[-800:]))' 2>/dev/null || echo '\"\"')}"
done
