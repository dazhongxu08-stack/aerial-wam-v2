"""Natural-language find task -> ``MissionSpec`` (rule based, Chinese + English).

"去前方200m左右找一个红色的凳子，然后把位置发给机器狗" ->
direction 0 rad (body), distance 200 m +-25 %, prompts ["stool", "bench", "chair"],
color "red", report_to "dog". Prompts carry no colour: a colour word in the text
prompt pulls YOLO-World toward any object of that colour; colour is checked on
the pixels instead.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# Body-relative bearings (rad, positive = right, matching yaw north->east).
_BODY_DIRS: List[Tuple[str, float]] = [
    ("左前方", -math.pi / 4), ("右前方", math.pi / 4), ("左后方", -3 * math.pi / 4), ("右后方", 3 * math.pi / 4),
    ("正前方", 0.0), ("前方", 0.0), ("前面", 0.0), ("往前", 0.0), ("向前", 0.0), ("前", 0.0),
    ("左侧", -math.pi / 2), ("左边", -math.pi / 2), ("左方", -math.pi / 2),
    ("右侧", math.pi / 2), ("右边", math.pi / 2), ("右方", math.pi / 2),
    ("后方", math.pi), ("后面", math.pi),
    ("front left", -math.pi / 4), ("front right", math.pi / 4),
    ("ahead", 0.0), ("forward", 0.0), ("in front", 0.0),
    ("to the left", -math.pi / 2), ("to the right", math.pi / 2), ("behind", math.pi),
]
# World headings (rad from north toward east).
_WORLD_DIRS: List[Tuple[str, float]] = [
    ("东北", math.pi / 4), ("西北", -math.pi / 4), ("东南", 3 * math.pi / 4), ("西南", -3 * math.pi / 4),
    ("正北", 0.0), ("北", 0.0), ("正东", math.pi / 2), ("东", math.pi / 2),
    ("正南", math.pi), ("南", math.pi), ("正西", -math.pi / 2), ("西", -math.pi / 2),
    ("northeast", math.pi / 4), ("northwest", -math.pi / 4), ("southeast", 3 * math.pi / 4),
    ("southwest", -3 * math.pi / 4), ("north", 0.0), ("east", math.pi / 2), ("south", math.pi),
    ("west", -math.pi / 2),
]

COLORS: Dict[str, str] = {
    "红": "red", "橙": "orange", "黄": "yellow", "绿": "green", "蓝": "blue", "紫": "purple",
    "白": "white", "黑": "black", "灰": "gray",
    "red": "red", "orange": "orange", "yellow": "yellow", "green": "green", "blue": "blue",
    "purple": "purple", "white": "white", "black": "black", "gray": "gray", "grey": "gray",
}

# Chinese noun -> (canonical label, open-vocab prompts). Longer keys first when matching.
OBJECTS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "凳子": ("stool", ("stool", "bench", "chair")),
    "板凳": ("stool", ("stool", "bench")),
    "长椅": ("bench", ("bench", "park bench")),
    "长凳": ("bench", ("bench", "park bench")),
    "椅子": ("chair", ("chair", "stool")),
    "桌子": ("table", ("table",)),
    "箱子": ("box", ("box", "crate")),
    "纸箱": ("box", ("cardboard box", "box")),
    "木箱": ("crate", ("wooden crate", "crate")),
    "桶": ("barrel", ("barrel", "bucket")),
    "垃圾桶": ("trash can", ("trash can", "garbage can")),
    "邮箱": ("mailbox", ("mailbox",)),
    "叉车": ("forklift", ("forklift",)),
    "托盘": ("pallet", ("pallet",)),
    "货架": ("shelf", ("shelf", "rack")),
    "卡车": ("truck", ("truck",)),
    "汽车": ("car", ("car",)),
    "车": ("car", ("car", "vehicle")),
    "自行车": ("bicycle", ("bicycle",)),
    "人": ("person", ("person",)),
    "背包": ("backpack", ("backpack", "bag")),
    "锥桶": ("traffic cone", ("traffic cone", "cone")),
    "雪糕筒": ("traffic cone", ("traffic cone", "cone")),
    "旗子": ("flag", ("flag",)),
    "灭火器": ("fire extinguisher", ("fire extinguisher",)),
}
_EN_OBJECTS = ("forklift", "pallet", "truck", "stool", "bench", "chair", "table", "box", "crate", "barrel",
               "trash can", "mailbox", "car",
               "bicycle", "person", "backpack", "traffic cone", "flag")

_DOG_WORDS = ("机器狗", "机械狗", "狗", "robot dog", "dog", "ugv")
# Report targets and the vehicle itself; their characters must not seed an object match.
_NOT_TARGET = ("机器狗", "机械狗", "机器人", "无人车", "无人机", "飞机")
_APPROX_WORDS = ("左右", "大约", "大概", "附近", "约", "about", "around", "roughly", "approximately", "~")


@dataclass
class MissionSpec:
    text: str
    label: str
    prompts: List[str]
    color: Optional[str] = None
    distance_m: Optional[float] = None
    tolerance_m: Optional[float] = None
    body_bearing: Optional[float] = None
    world_heading: Optional[float] = None
    report_to: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ParseError(ValueError):
    pass


def _distance(t: str) -> Tuple[Optional[float], bool]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(公里|千米|km|米|meters?|metres?|m)(?![a-z])", t, re.I)
    if not m:
        return None, False
    v = float(m.group(1)) * (1000.0 if m.group(2).lower() in ("公里", "千米", "km") else 1.0)
    window = t[max(0, m.start() - 4): m.end() + 4]
    return v, any(w in window for w in _APPROX_WORDS)


def _first(t: str, table: List[Tuple[str, float]]) -> Optional[float]:
    best: Optional[Tuple[int, int, float]] = None
    for k, v in table:
        i = t.find(k)
        if i >= 0 and (best is None or (i, -len(k)) < (best[0], best[1])):
            best = (i, -len(k), v)
    return None if best is None else best[2]


def parse_instruction(text: str, *, default_tolerance: float = 0.25) -> MissionSpec:
    t = text.strip()
    low = t.lower()
    color = None
    for k in sorted(COLORS, key=len, reverse=True):
        if k in low:
            color = COLORS[k]
            break
    t_obj = t
    for w in _NOT_TARGET:
        t_obj = t_obj.replace(w, " ")
    label, prompts = None, ()
    for k in sorted(OBJECTS, key=len, reverse=True):
        if k in t_obj:
            label, prompts = OBJECTS[k]
            break
    if label is None:
        for k in _EN_OBJECTS:
            if re.search(rf"\b{k}s?\b", low):
                label, prompts = k, (k,)
                break
    if label is None:
        m = re.search(r"(?:find|look for|search for)\s+(?:an?\s+|the\s+)?(?:\w+\s+)?([a-z][a-z ]{2,30})", low)
        if not m:
            raise ParseError(f"no target object found in: {text!r}")
        label = m.group(1).strip()
        if color and label.startswith(color + " "):
            label = label[len(color) + 1:]
        prompts = (label,)
    dist, approx = _distance(low)
    tol = None
    if dist is not None:
        tol = max(15.0, dist * (default_tolerance if approx else 0.15))
    world = _first(low.replace("东西", " "), _WORLD_DIRS)
    body = None if world is not None else _first(low, _BODY_DIRS)
    if dist is not None and body is None and world is None:
        body = 0.0
    report = ["dog"] if any(w in low for w in _DOG_WORDS) else []
    return MissionSpec(text=t, label=label, prompts=list(dict.fromkeys(prompts)), color=color, distance_m=dist,
                       tolerance_m=tol, body_bearing=body, world_heading=world, report_to=report)
