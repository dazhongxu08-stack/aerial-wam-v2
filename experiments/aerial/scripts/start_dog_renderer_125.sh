#!/usr/bin/env bash
# Second AirVLN renderer on 125 for the dog stand-in (camera above a low body), API on :41481.
# Leaves the :41451 renderer and its settings.json alone. Stop with: kill "$(cat $DOG_DIR/renderer.pid)"
set -euo pipefail
REPO="${REPO:-/home/yao/aerial-wam-v2}"
BIN_DIR="${BIN_DIR:-/home/yao/aerial_airsim_persistent/scene/env_airsim_16/LinuxNoEditor/AirVLN/Binaries/Linux}"
DOG_DIR="${DOG_DIR:-/home/yao/dog_airsim}"
PORT="${PORT:-41481}"
PY="${PY:-/home/yao/sim_verify/.venv/bin/python}"
mkdir -p "$DOG_DIR"
if [[ -f "$DOG_DIR/renderer.pid" ]] && kill -0 "$(cat "$DOG_DIR/renderer.pid")" 2>/dev/null; then
  echo "dog renderer already running pid=$(cat "$DOG_DIR/renderer.pid")"; exit 0
fi
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT already in use; pick another PORT"; exit 1; fi
"$PY" "$REPO/experiments/aerial/scripts/make_dog_airsim_settings.py" \
  --base "$BIN_DIR/settings.json" --out "$DOG_DIR/settings_dog.json" --port "$PORT" --cam-z -0.15
gcc -shared -fPIC -O2 -o "$DOG_DIR/port_remap_shim.so" "$REPO/experiments/aerial/scripts/port_remap_shim.c" -ldl
cd "$BIN_DIR"
PORT_REMAP_FROM=41451 PORT_REMAP_TO="$PORT" LD_PRELOAD="$DOG_DIR/port_remap_shim.so" \
  setsid nohup ./AirVLN-Linux-Shipping AirVLN -windowed -RenderOffScreen -nosound -Vulkan -ResX=640 -ResY=360 \
  -settings="$DOG_DIR/settings_dog.json" >"$DOG_DIR/renderer.log" 2>&1 </dev/null &
echo $! >"$DOG_DIR/renderer.pid"
for _ in $(seq 60); do
  if ss -ltnp 2>/dev/null | grep ":$PORT " | grep -q "pid=$(cat "$DOG_DIR/renderer.pid"),"; then echo "dog renderer up pid=$(cat "$DOG_DIR/renderer.pid") port=$PORT"; exit 0; fi
  kill -0 "$(cat "$DOG_DIR/renderer.pid")" 2>/dev/null || { tail -20 "$DOG_DIR/renderer.log"; exit 1; }
  sleep 2
done
echo "timeout waiting for :$PORT"; exit 1
