#!/usr/bin/env bash
# Build (optional) + install H12 Tello GS app, reverse Orin console to H12.
# Sticks (RCSDK channels) control Tello; video from Orin JPEG.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
APP_DIR="${ROOT}/experiments/aerial/h12_tello_console"
ADB="${ADB:-$HOME/sdk/android/platform-tools/adb}"
PORT="${PORT:-8090}"
BUILD="${BUILD:-1}"

if [[ ! -x "${ADB}" ]]; then
  echo "ERROR: adb not found at ${ADB}" >&2
  exit 1
fi

if [[ "${BUILD}" == "1" ]]; then
  export JAVA_HOME="${JAVA_HOME:-$HOME/sdk/jdk-17/Contents/Home}"
  export ANDROID_HOME="${ANDROID_HOME:-$HOME/sdk/android}"
  (cd "${APP_DIR}" && ./gradlew assembleDebug --no-daemon)
fi

APK="${APP_DIR}/app/build/outputs/apk/debug/app-debug.apk"
[[ -f "${APK}" ]] || { echo "missing ${APK} (BUILD=1 to compile)"; exit 1; }

echo "Force-stop Skydroid / Lumos apps that steal RCSDK ports"
"${ADB}" shell am force-stop com.skydroid.h12tool >/dev/null 2>&1 || true
"${ADB}" shell am force-stop com.skydroid.fpvtest >/dev/null 2>&1 || true
"${ADB}" shell am force-stop com.aerial.ludgs >/dev/null 2>&1 || true

echo "Install ${APK}"
"${ADB}" install -r -d "${APK}"
# Drop saved URL so default is Mac adb reverse 127.0.0.1:8090
"${ADB}" shell pm clear com.aerial.tellogs >/dev/null 2>&1 || true
"${ADB}" reverse --remove-all >/dev/null 2>&1 || true
"${ADB}" reverse "tcp:${PORT}" "tcp:${PORT}"

"${ADB}" shell am start -n com.aerial.tellogs/.MainActivity --es base_url "http://127.0.0.1:${PORT}"
echo "H12 Tello GS launched. URL http://127.0.0.1:${PORT} (adb reverse → Mac). Orin is Ethernet-only."
echo "MODE2: ch1-4 sticks → Tello; ch5 emergency; ch6 takeoff/land. Close H12助手 / Fly first."
