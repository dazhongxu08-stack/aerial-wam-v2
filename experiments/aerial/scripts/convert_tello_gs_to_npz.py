#!/usr/bin/env python3
"""CLI wrapper: tello_gs JPEG+jsonl → schema-v2 episode_*.npz."""
from experiments.aerial.rl.convert_tello_gs import main

if __name__ == "__main__":
    raise SystemExit(main())
