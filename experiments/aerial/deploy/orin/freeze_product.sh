#!/usr/bin/env bash
# Strip plaintext sources from an installed /opt/aerial_orin tree.
# Keeps .pyc + weights + config + ELF + shell entrypoints.
set -euo pipefail
DEST="${1:-/opt/aerial_orin}"
PY="${PY:-python3}"

[[ -d "$DEST/aerial_finder" ]] || { echo "missing $DEST" >&2; exit 1; }

echo "[freeze] compileall $DEST"
"$PY" -m compileall -q -b -x 'weights/parser' "$DEST"

echo "[freeze] drop plaintext code and docs"
# Runtime python → bytecode only
find "$DEST" \( -name '*.py' -o -name '*.pyi' \) -type f -delete
# Human-readable docs / VCS / C++ sources (keep compiled ELF under lumos_sdk/build)
find "$DEST" \( -name '*.md' -o -name '*.rst' -o -name '.gitignore' \
  -o -name 'CMakeLists.txt' -o -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \
  -o -name '*.h' -o -name '*.hpp' -o -name '*.c' \) -type f -delete
rm -rf "$DEST/aerial_finder/tests" "$DEST/aerial_finder/tools" "$DEST/aerial_finder/deploy" \
  "$DEST/experiments/aerial/rl/tests" "$DEST/experiments/aerial/rl/.claude" \
  "$DEST/experiments/aerial/rl/logs" "$DEST/experiments/aerial/rl/artifacts" \
  "$DEST/lumos_sdk/src" "$DEST/lumos_sdk/example" "$DEST/lumos_sdk/docs" \
  "$DEST/lumos_sdk/scripts" "$DEST/weights/m_p/README.md" \
  "$DEST/weights/m_p/LICENSE" || true

# Entry scripts must run .pyc (legacy or current names)
for b in h12-lud-bridge rt-link; do
  [[ -f "$DEST/bin/$b" ]] && sed -i 's|h12_lud_bridge.py|h12_lud_bridge.pyc|' "$DEST/bin/$b" || true
done
for b in tello-orin rt-air; do
  [[ -f "$DEST/bin/$b" ]] && sed -i 's|tello_consequence_collect.py|tello_consequence_collect.pyc|' "$DEST/bin/$b" || true
done

echo "[freeze] remaining text-like files:"
find "$DEST" \( -name '*.py' -o -name '*.md' -o -name '*.cpp' \) | head || echo "(none)"
echo "[freeze] done $(du -sh "$DEST" | awk '{print $1}')"
