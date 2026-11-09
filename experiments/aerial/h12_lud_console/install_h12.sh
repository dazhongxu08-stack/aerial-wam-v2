#!/usr/bin/env bash
# Build + install H12 Lumos dog console. Close H12助手 / Skydroid Fly first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
APP_DIR="${ROOT}/experiments/aerial/h12_lud_console"
ADB="${ADB:-$HOME/sdk/android/platform-tools/adb}"
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
[[ -f "${APK}" ]] || { echo "missing ${APK}"; exit 1; }

echo "Force-stop Skydroid tools that steal RCSDK ports"
"${ADB}" shell am force-stop com.skydroid.h12tool >/dev/null 2>&1 || true
"${ADB}" shell am force-stop com.skydroid.fpvtest >/dev/null 2>&1 || true
"${ADB}" shell am force-stop com.aerial.tellogs >/dev/null 2>&1 || true

echo "Install ${APK}"
"${ADB}" install -r -d "${APK}"
"${ADB}" reverse --remove-all >/dev/null 2>&1 || true
"${ADB}" shell am start -n com.aerial.ludgs/.MainActivity
echo "Launched Lumos Dog. Default H12 → http://172.20.10.4:8092 (Orin) → dog."
echo "On Orin: python3 /home/ubuntu/h12_lud_bridge.py --port 8092"
