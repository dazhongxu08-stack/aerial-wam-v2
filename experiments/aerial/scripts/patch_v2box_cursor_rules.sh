#!/usr/bin/env bash
# Patch V2BOX leaf.conf so Cursor domains use real DNS + Japan proxy.
# Run after V2BOX VPN is connected, then reconnect VPN once from the app.

set -euo pipefail

LEAF_DIR="${HOME}/Library/Group Containers/group.hossin.asaadi.V2Box"
LEAF_CONF="${LEAF_DIR}/leaf.conf"
PREFS_DOMAIN="hossin.asaadi.V2Box"

if [[ ! -f "${LEAF_CONF}" ]]; then
  echo "leaf.conf not found. Is V2BOX installed?"
  exit 1
fi

chflags nouchg "${LEAF_CONF}" 2>/dev/null || true

TUNFD="$(awk '/^tun-fd/{print $3; exit}' "${LEAF_CONF}")"
if [[ -z "${TUNFD}" ]]; then
  TUNFD="5"
fi

cat > "${LEAF_CONF}" <<EOF
[General]
loglevel = none
always-fake-ip = *
always-real-ip = cursor.sh,cursor.com,api2.cursor.sh,api3.cursor.sh,anthropic.com,openai.com,claude.ai,chatgpt.com

tun-fd = ${TUNFD}



[Env]        
ENABLE_IPV6 = 0
PREFER_IPV6 = 0

[Proxy]
Socks = socks,  127.0.0.1, 55486


[Rule]
DOMAIN-SUFFIX,cursor.sh,Socks
DOMAIN-SUFFIX,cursor.com,Socks
DOMAIN-KEYWORD,cursor,Socks
DOMAIN-SUFFIX,anthropic.com,Socks
DOMAIN-SUFFIX,openai.com,Socks
DOMAIN-SUFFIX,claude.ai,Socks
DOMAIN-SUFFIX,chatgpt.com,Socks
NETWORK, tcp, Socks
NETWORK, udp, Socks
FINAL, direct
EOF

defaults write "${PREFS_DOMAIN}" enableRouting -bool true
defaults write "${PREFS_DOMAIN}" routingProxyDomains -array \
  "cursor.sh" "cursor.com" "anthropic.com" "openai.com" "claude.ai" "chatgpt.com"
defaults write "${PREFS_DOMAIN}" userRules -string $'DOMAIN-SUFFIX,cursor.sh,Socks\nDOMAIN-SUFFIX,cursor.com,Socks\nDOMAIN-KEYWORD,cursor,Socks\nDOMAIN-SUFFIX,anthropic.com,Socks\nDOMAIN-SUFFIX,openai.com,Socks\nDOMAIN-SUFFIX,claude.ai,Socks\nDOMAIN-SUFFIX,chatgpt.com,Socks'

echo "Patched ${LEAF_CONF} (tun-fd=${TUNFD})"
echo "Reconnect V2BOX VPN once (off -> on), then run:"
echo "  dig +short api2.cursor.sh"
echo "Expected: real AWS IPs, not 198.18.x.x"
