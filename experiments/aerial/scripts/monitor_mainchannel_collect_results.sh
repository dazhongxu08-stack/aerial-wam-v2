#!/usr/bin/env bash
exec python3 "$(cd "$(dirname "$0")" && pwd)/monitor_mainchannel_collect_results.py" "$@"
