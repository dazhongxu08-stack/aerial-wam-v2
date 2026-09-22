"""HJ USB 2.0 Camera (Microdia 0c45:636b) V4L2 presets for Orin bench / deploy.

Uses ``v4l2-ctl`` on Linux; no-op with a warning elsewhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

from experiments.aerial.deploy.c922_controls import (
    apply_v4l2_controls,
    read_v4l2_controls,
    v4l2_device_path,
)

logger = logging.getLogger(__name__)

PRESET_OUTDOOR_BENCH = "outdoor_bench"
PRESET_OUTDOOR_BRIGHT = "outdoor_bright"
PRESET_OUTDOOR_DIM = "outdoor_dim"


@dataclass(frozen=True)
class HJPreset:
    name: str
    auto_exposure: int = 1
    exposure_time_absolute: int = 6
    brightness: int = -10
    gain: int = 0
    contrast: int = 32
    saturation: int = 32
    hue: int = -30
    white_balance_automatic: int = 1
    backlight_compensation: int = 0
    exposure_dynamic_framerate: int = 0

    def controls(self) -> Dict[str, int]:
        return {
            "auto_exposure": int(self.auto_exposure),
            "exposure_time_absolute": int(self.exposure_time_absolute),
            "brightness": int(self.brightness),
            "gain": int(self.gain),
            "contrast": int(self.contrast),
            "saturation": int(self.saturation),
            "hue": int(self.hue),
            "white_balance_automatic": int(self.white_balance_automatic),
            "backlight_compensation": int(self.backlight_compensation),
            "exposure_dynamic_framerate": int(self.exposure_dynamic_framerate),
        }


_PRESETS: Dict[str, HJPreset] = {
    PRESET_OUTDOOR_BENCH: HJPreset(
        name=PRESET_OUTDOOR_BENCH,
        exposure_time_absolute=6,
        brightness=-10,
        hue=-30,
        saturation=32,
    ),
    PRESET_OUTDOOR_BRIGHT: HJPreset(
        name=PRESET_OUTDOOR_BRIGHT,
        exposure_time_absolute=5,
        brightness=-10,
        hue=-30,
        saturation=32,
    ),
    PRESET_OUTDOOR_DIM: HJPreset(
        name=PRESET_OUTDOOR_DIM,
        exposure_time_absolute=8,
        brightness=-5,
        hue=-25,
        saturation=36,
    ),
}


def list_presets() -> List[str]:
    return sorted(_PRESETS.keys())


def get_preset(name: str) -> HJPreset:
    key = str(name).strip()
    if key not in _PRESETS:
        raise KeyError(f"unknown HJ preset {key!r}; choose from {list_presets()}")
    return _PRESETS[key]


def apply_preset(device: str, preset_name: str = PRESET_OUTDOOR_BENCH, *, dry_run: bool = False) -> HJPreset:
    preset = get_preset(preset_name)
    apply_v4l2_controls(device, preset.controls(), dry_run=dry_run)
    if not dry_run:
        dev_path = v4l2_device_path(device)
        logger.info(
            "HJ preset %s applied on %s: %s",
            preset.name,
            dev_path,
            ", ".join(f"{k}={v}" for k, v in preset.controls().items()),
        )
    return preset


def read_controls(device: str, preset_name: str = PRESET_OUTDOOR_BENCH) -> Dict[str, int | None]:
    preset = get_preset(preset_name)
    return read_v4l2_controls(device, list(preset.controls().keys()))
