#!/usr/bin/env bash
# AX88179B（en7）接到机器人 ms-thor 的交换机。
# 对端是 eth1：192.168.1.100/24。不要改机器人，也不要占用 .101 / .102。
# 路由器保持 0.0.0.0，办公网仍走 Wi-Fi。
set -euo pipefail

SERVICE="AX88179B"
IP="192.168.1.50"
MASK="255.255.255.0"
ROUTER="0.0.0.0"
TARGET="192.168.1.100"

networksetup -setmanual "${SERVICE}" "${IP}" "${MASK}" "${ROUTER}"
echo "AX88179B → 机器人 ${TARGET}（user@${TARGET}）"
networksetup -getinfo "${SERVICE}"
