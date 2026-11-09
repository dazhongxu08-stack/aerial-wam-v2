#!/usr/bin/env bash
# Install a packed tree into /opt/aerial_finder (or DEST) and enable systemd.
#
#   sudo ./install.sh [/opt/aerial_finder]
# If already extracted to /opt/aerial_finder, run from there:
#   sudo /opt/aerial_finder/deploy/install.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
# Pack layout: $ROOT/{aerial_finder,bin,weights,config.json,deploy,venv?}
ROOT=$(cd "$HERE/.." && pwd)
DEST=${1:-/opt/aerial_finder}

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run as root: sudo $0 $*" >&2
  exit 1
fi

if [[ "$ROOT" != "$DEST" ]]; then
  echo "[install] sync $ROOT -> $DEST"
  mkdir -p "$DEST"
  rsync -a --delete \
    --exclude venv --exclude runs --exclude '__pycache__' \
    "$ROOT/" "$DEST/"
fi

mkdir -p "$DEST/runs" "$DEST/bin"
if [[ ! -x "$DEST/bin/aerial-finder" ]]; then
  install -m 0755 "$HERE/bin/aerial-finder" "$DEST/bin/aerial-finder"
fi
chmod 0755 "$DEST/bin/aerial-finder"

# venv + pip (reuses JetPack torch via --system-site-packages)
bash "$DEST/deploy/install_venv.sh" "$DEST"

# systemd unit with absolute paths
UNIT=/etc/systemd/system/aerial-finder.service
sed -e "s|/opt/aerial_finder|$DEST|g" "$DEST/deploy/aerial-finder.service" > "$UNIT"
systemctl daemon-reload
echo "[install] unit written to $UNIT"
echo "[install] start: systemctl enable --now aerial-finder"
echo "[install] smoke:  curl -s http://127.0.0.1:8088/api/v1/health"
echo "[install] parse:  $DEST/bin/aerial-finder parse '正前方10米找一个红色的凳子'"
