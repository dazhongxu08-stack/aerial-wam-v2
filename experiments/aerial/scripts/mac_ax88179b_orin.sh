#!/usr/bin/env bash
# 把 AX88179B（en7）换回当前的 Orin 有线设置。
# 2026-10-26 记录：192.168.54.2/16，路由器 0.0.0.0。
# Orin 地址 192.168.54.111，SSH 别名 orin-eth。
set -euo pipefail

SERVICE="AX88179B"
IP="192.168.54.2"
MASK="255.255.0.0"
ROUTER="0.0.0.0"
TARGET="192.168.54.111"

networksetup -setmanual "${SERVICE}" "${IP}" "${MASK}" "${ROUTER}"
echo "AX88179B → Orin ${TARGET}（ssh orin-eth）"
networksetup -getinfo "${SERVICE}"
