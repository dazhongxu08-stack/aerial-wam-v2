#!/usr/bin/env python3
"""Run tello_find_object unchanged against lud_tello_shim (the dog), with the dog camera's FOV.

  PYTHONPATH=. python3 experiments/aerial/scripts/lud_find_object.py --hfov-deg 110 -- \\
    --base http://127.0.0.1:8090 --model ~/yw_forklift.torchscript --classes forklift,... ...
"""
from __future__ import annotations

import sys

import experiments.aerial.scripts.tello_find_object as tfo


def main() -> int:
    argv = sys.argv[1:]
    if "--hfov-deg" in argv:
        i = argv.index("--hfov-deg")
        tfo.HFOV_DEG = float(argv[i + 1])
        del argv[i:i + 2]
    if argv and argv[0] == "--":
        argv = argv[1:]
    sys.argv = [sys.argv[0]] + argv
    print(f"HFOV_DEG={tfo.HFOV_DEG}", flush=True)
    return tfo.main()


if __name__ == "__main__":
    raise SystemExit(main())
