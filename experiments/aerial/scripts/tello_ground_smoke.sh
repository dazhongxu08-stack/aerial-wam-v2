#!/usr/bin/env bash
# Orin Tello ground smoke test (no takeoff).
# Checks Wi-Fi, SDK, telemetry freshness, ground-lock RC, and camera frames.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${ORIN_PY:-$HOME/sim_verify/.venv/bin/python}"
CONSOLE_URL="${CONSOLE_URL:-http://127.0.0.1:8090}"
LOCAL_IP="${LOCAL_IP:-192.168.10.2}"
TELLO_IP="${TELLO_IP:-192.168.10.1}"
PASS=0
FAIL=0
WARN=0

pass() { echo "PASS  $*"; PASS=$((PASS+1)); }
fail() { echo "FAIL  $*"; FAIL=$((FAIL+1)); }
warn() { echo "WARN  $*"; WARN=$((WARN+1)); }

echo "=== Tello ground smoke $(date -Is) ==="

echo "--- A. Wi-Fi ---"
# Prefer any iface with 192.168.10.x (USB dongle or onboard).
WIFI_IFACE="$(ip -4 -o addr show | awk '/192\.168\.10\./{print $2; exit}')"
WIFI_IFACE="${WIFI_IFACE:-wlP1p1s0}"
SSID="$(nmcli -t -f GENERAL.CONNECTION device show "${WIFI_IFACE}" 2>/dev/null | head -1 || true)"
SSID="${SSID#GENERAL.CONNECTION:}"
STATE="$(nmcli -t -f GENERAL.STATE device show "${WIFI_IFACE}" 2>/dev/null | head -1 || true)"
ADDR="$(ip -4 -o addr show "${WIFI_IFACE}" 2>/dev/null | awk '{print $4}' | head -1 || true)"
echo "  iface=${WIFI_IFACE} ssid=${SSID:-none} state=${STATE:-none} addr=${ADDR:-none}"
if [[ "${SSID}" == TELLO* ]] && [[ "${ADDR}" == 192.168.10.* ]]; then
  pass "wifi associated to Tello with 192.168.10.x"
elif [[ "${ADDR}" == 192.168.10.* ]]; then
  pass "wifi has Tello subnet on ${WIFI_IFACE} (ssid=${SSID:-none})"
else
  fail "wifi not on Tello (iface=${WIFI_IFACE} ssid=${SSID:-none} addr=${ADDR:-none})"
fi

echo "--- B. Console HTTP ---"
if curl -fsS -m3 "${CONSOLE_URL}/api/status" >/tmp/tello_smoke_status.json; then
  pass "console /api/status reachable"
else
  fail "console /api/status unreachable at ${CONSOLE_URL}"
  echo "RESULT pass=$PASS fail=$FAIL warn=$WARN"
  exit 1
fi

echo "--- C. SDK / telemetry ---"
"$PY" - <<'PY' | tee /tmp/tello_smoke_telem.txt
import json, time, urllib.request
url = "http://127.0.0.1:8090/api/status"
def st():
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.load(r)
rows = []
for _ in range(5):
    rows.append(st()); time.sleep(0.4)
s = rows[-1]
print(json.dumps({
  "ok": s.get("ok"), "battery": s.get("battery"), "temp": s.get("temp"),
  "state_age_s": s.get("state_age_s"), "yaw": s.get("yaw"), "tof": s.get("tof"),
  "height_cm": s.get("height_cm"), "airborne": s.get("airborne"),
  "allow_fly": s.get("allow_fly"),
}, indent=2))
open("/tmp/tello_smoke_telem.env","w").write(
  f"AGE={s.get('state_age_s')}\nBAT={s.get('battery')}\nTEMP={s.get('temp')}\n"
  f"AIR={s.get('airborne')}\nFLY={s.get('allow_fly')}\n"
)
PY
# shellcheck disable=SC1091
source /tmp/tello_smoke_telem.env
if [[ -n "${AGE}" && "${AGE}" != "None" ]] && "$PY" -c "import sys; sys.exit(0 if float('${AGE}')<3 else 1)"; then
  pass "telemetry fresh (state_age_s=${AGE})"
else
  fail "telemetry stale (state_age_s=${AGE:-none})"
fi
if [[ -n "${BAT}" && "${BAT}" != "None" ]] && [[ "${BAT}" -ge 15 ]]; then
  pass "battery ${BAT}% (>=15%)"
elif [[ -n "${BAT}" && "${BAT}" != "None" ]]; then
  warn "battery low ${BAT}%"
else
  fail "battery unknown"
fi
if [[ -n "${TEMP}" && "${TEMP}" != "None" ]] && [[ "${TEMP}" -ge 100 ]]; then
  warn "temperature high ${TEMP}C (AP drop risk)"
elif [[ -n "${TEMP}" && "${TEMP}" != "None" ]]; then
  pass "temperature ${TEMP}C"
fi
if [[ "${AIR}" == "False" || "${AIR}" == "false" ]]; then
  pass "airborne=false (ground)"
else
  fail "airborne unexpected: ${AIR}"
fi
if [[ "${FLY}" == "False" || "${FLY}" == "false" ]]; then
  pass "allow_fly=false (ground lock)"
else
  warn "allow_fly=true"
fi

echo "--- D. Control lock ---"
# non-zero RC must be rejected; takeoff must be rejected
RC_CODE=$(curl -sS -m3 -o /tmp/tello_smoke_rc.json -w "%{http_code}" \
  -H 'Content-Type: application/json' \
  -d '{"a":0,"b":40,"c":0,"d":0}' "${CONSOLE_URL}/api/rc" || echo 000)
TO_CODE=$(curl -sS -m3 -o /tmp/tello_smoke_to.json -w "%{http_code}" \
  -H 'Content-Type: application/json' \
  -d '{"cmd":"takeoff"}' "${CONSOLE_URL}/api/cmd" || echo 000)
ZERO_CODE=$(curl -sS -m3 -o /tmp/tello_smoke_zero.json -w "%{http_code}" \
  -H 'Content-Type: application/json' \
  -d '{"a":0,"b":0,"c":0,"d":0}' "${CONSOLE_URL}/api/rc" || echo 000)
echo "  rc_nonzero_http=${RC_CODE} body=$(cat /tmp/tello_smoke_rc.json 2>/dev/null || true)"
echo "  takeoff_http=${TO_CODE} body=$(cat /tmp/tello_smoke_to.json 2>/dev/null || true)"
echo "  rc_zero_http=${ZERO_CODE} body=$(cat /tmp/tello_smoke_zero.json 2>/dev/null || true)"
if [[ "${RC_CODE}" == "403" ]]; then pass "non-zero RC blocked"; else fail "non-zero RC not blocked (http ${RC_CODE})"; fi
if [[ "${TO_CODE}" == "403" ]]; then pass "takeoff blocked"; else fail "takeoff not blocked (http ${TO_CODE})"; fi
if [[ "${ZERO_CODE}" == "200" ]]; then pass "zero RC accepted"; else warn "zero RC http ${ZERO_CODE}"; fi

echo "--- E. Camera / frames (8s) ---"
"$PY" - <<'PY' || CAM_RC=$?
import json, time, urllib.request, hashlib, struct
import sys

def status():
    with urllib.request.urlopen("http://127.0.0.1:8090/api/status", timeout=3) as r:
        return json.load(r)

def snap():
    with urllib.request.urlopen("http://127.0.0.1:8090/snapshot.jpg", timeout=5) as r:
        return r.read()

t0 = time.time()
s0 = status()
seq0 = int(s0.get("frame_seq") or 0)
hashes = []
sizes = []
while time.time() - t0 < 8.0:
    try:
        jpg = snap()
        sizes.append(len(jpg))
        hashes.append(hashlib.md5(jpg).hexdigest())
    except Exception as e:
        print("snap_err", e)
    time.sleep(0.4)

s1 = status()
seq1 = int(s1.get("frame_seq") or 0)
uniq = len(set(hashes))
dt = max(1e-3, time.time() - t0)
delta = max(0, seq1 - seq0)
fps = delta / dt
w = h = None
if sizes:
    data = snap()
    i = data.find(b"\xff\xc0")
    if i < 0:
        i = data.find(b"\xff\xc2")
    if i >= 0 and i + 9 < len(data):
        h, w = struct.unpack(">HH", data[i+5:i+9])

print(json.dumps({
    "frame_seq0": seq0,
    "frame_seq1": seq1,
    "seq_delta": delta,
    "window_s": round(dt, 2),
    "publish_fps": round(fps, 2),
    "unique_jpeg_hashes": uniq,
    "jpeg_bytes_min": min(sizes) if sizes else None,
    "jpeg_bytes_max": max(sizes) if sizes else None,
    "jpeg_wh": [w, h],
    "frame_age_s": s1.get("frame_age_s"),
}, indent=2))

ok_seq = delta >= 5
ok_var = uniq >= 3
ok_res = (w is not None and h is not None and min(w, h) >= 240)
print("CAM_SEQ_OK", ok_seq, delta)
print("CAM_VAR_OK", ok_var, uniq)
print("CAM_RES_OK", ok_res, w, h)
sys.exit(0 if (ok_seq and ok_var and ok_res) else 2)
PY
CAM_RC=${CAM_RC:-0}
if [[ "${CAM_RC}" -eq 0 ]]; then
  pass "camera frames advancing with content change + usable resolution"
elif [[ "${CAM_RC}" -eq 2 ]]; then
  fail "camera frames do not meet smoke bar (fps/variety/resolution)"
else
  fail "camera probe errored (rc=${CAM_RC})"
fi

echo "--- F. Inference camera path (TelloStreamCamera) ---"
# If console already owns UDP 11111, skip exclusive open and rely on section E.
if ss -ulnp 2>/dev/null | grep -q ':11111' || pgrep -f '[t]ello_ground_console.py' >/dev/null 2>&1; then
  warn "UDP 11111 / console in use — skip exclusive TelloStreamCamera open; UI frames cover feed check"
else
"$PY" - <<PY
import os, sys, time, json
sys.path.insert(0, "${ROOT}")
from experiments.aerial.deploy.tello_camera import TelloStreamCamera
cam = TelloStreamCamera(port=11111, open_timeout_s=20.0, stale_restart_s=1.5, restart_cooldown_s=3.0)
try:
    cam.open()
except Exception as e:
    print("INFER_CAM_OPEN FAIL", type(e).__name__, e)
    raise SystemExit(3)
t0 = time.time()
n = 0
last = -1
shape = None
try:
    while time.time() - t0 < 8.0:
        fr = cam.read_bgr_new()
        if fr is not None and cam.frame_seq != last:
            last = cam.frame_seq
            n += 1
            shape = list(fr.shape)
        time.sleep(0.05)
finally:
    cam.close()
dt = time.time() - t0
print(json.dumps({"new_frames": n, "fps": round(n/dt, 2), "last_seq": last, "shape": shape}, indent=2))
print("INFER_CAM_OK", n >= 5)
raise SystemExit(0 if n >= 5 else 4)
PY
INF_RC=$?
if [[ "${INF_RC}" -eq 0 ]]; then
  pass "TelloStreamCamera produces fresh frames"
elif [[ "${INF_RC}" -eq 3 ]]; then
  warn "TelloStreamCamera open failed"
elif [[ "${INF_RC}" -eq 4 ]]; then
  fail "TelloStreamCamera open but too few fresh frames"
else
  fail "TelloStreamCamera probe rc=${INF_RC}"
fi
fi

echo "=== RESULT pass=${PASS} fail=${FAIL} warn=${WARN} ==="
if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi
exit 0
