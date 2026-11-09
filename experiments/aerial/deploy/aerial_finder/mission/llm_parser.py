"""Lightweight Qwen2.5-0.5B instruction parser. Rules remain the numeric/color/dir source.

Load is lazy. Failures fall back to ``parse_instruction`` (no exception at serve start
beyond a log). The model is not on the 5 Hz control path.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .parser import (
    COLORS,
    OBJECTS,
    MissionSpec,
    ParseError,
    _distance,
    parse_instruction,
)

log = logging.getLogger("aerial_finder.parser")

_COLOR_VALS = set(COLORS.values())
_EN_TO_PROMPTS: dict = {}
for _label, _prompts in OBJECTS.values():
    if _label not in _EN_TO_PROMPTS or len(_prompts) > len(_EN_TO_PROMPTS[_label]):
        _EN_TO_PROMPTS[_label] = _prompts
_JSON_RE = re.compile(r"\{.*\}", re.S)


SYSTEM = (
    "Extract a robot find-object task as JSON only. Keys:\n"
    'label: short English object name (stool, chair, person, box, car, ...)\n'
    "prompts: 1-3 English detector phrases, no colour words\n"
    "color: red|orange|yellow|green|blue|purple|white|black|gray or null\n"
    "distance_m: meters as a number or null\n"
    "body_bearing: radians, 0 forward, positive right, or null\n"
    "world_heading: radians from north toward east, or null\n"
    'report_to: ["dog"] if a robot dog should be notified, else []\n'
    "No markdown."
)


def _extract_json(text: str) -> Optional[dict]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    m = _JSON_RE.search(t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _prompts_for(label: str, raw_prompts: Any) -> list:
    key = (label or "").strip().lower()
    if key in _EN_TO_PROMPTS:
        return list(_EN_TO_PROMPTS[key])
    out = []
    if isinstance(raw_prompts, (list, tuple)):
        for p in raw_prompts:
            s = str(p).strip().lower()
            if not s or s in _COLOR_VALS or s in out or len(s) > 32:
                continue
            if s.startswith(("find ", "look ", "search ")):
                continue
            out.append(s)
    if not out and key:
        out = [key]
    return out[:3]


def spec_from_llm_json(text: str, obj: dict, *, default_tolerance: float = 0.25) -> Optional[MissionSpec]:
    label = str(obj.get("label") or "").strip().lower()
    if not label:
        return None
    color = obj.get("color")
    if isinstance(color, str):
        color = COLORS.get(color.strip().lower(), color.strip().lower())
        if color not in _COLOR_VALS:
            color = None
    else:
        color = None
    for cname in sorted(_COLOR_VALS, key=len, reverse=True):
        if label.startswith(cname + " "):
            label = label[len(cname) + 1:].strip()
            color = color or cname
            break
        if label.endswith(" " + cname):
            label = label[: -(len(cname) + 1)].strip()
            color = color or cname
            break
    for k, (canon, _) in OBJECTS.items():
        if k in label or canon == label:
            label = canon
            break
    else:
        for en in sorted(_EN_TO_PROMPTS, key=len, reverse=True):
            if en in label:
                label = en
                break
    if not label:
        return None
    dist, approx = _distance(text.lower())
    if dist is None:
        try:
            dist = float(obj["distance_m"]) if obj.get("distance_m") is not None else None
        except (TypeError, ValueError):
            dist = None
    tol = None
    if dist is not None:
        tol = max(15.0, dist * (default_tolerance if approx else 0.15))
    def _ang(key: str) -> Optional[float]:
        v = obj.get(key)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    report = obj.get("report_to") or []
    if not isinstance(report, list):
        report = []
    report = ["dog"] if any(str(x).lower() in ("dog", "机器狗", "robot dog") for x in report) else []
    return MissionSpec(
        text=text.strip(),
        label=label,
        prompts=_prompts_for(label, obj.get("prompts")),
        color=color,
        distance_m=dist,
        tolerance_m=tol,
        body_bearing=_ang("body_bearing"),
        world_heading=_ang("world_heading"),
        report_to=report,
    )


def merge_rule_llm(text: str, llm_spec: Optional[MissionSpec], *, default_tolerance: float = 0.25) -> MissionSpec:
    """Prefer rule fields when the keyword parser succeeded; LLM fills missing object/slots."""
    rule = None
    rule_err: Optional[ParseError] = None
    try:
        rule = parse_instruction(text, default_tolerance=default_tolerance)
    except ParseError as exc:
        rule_err = exc
    if rule is not None and llm_spec is None:
        return rule
    if rule is None and llm_spec is None:
        raise rule_err or ParseError(f"no target object found in: {text!r}")
    if rule is None:
        return llm_spec
    if llm_spec is None:
        return rule
    return MissionSpec(
        text=text.strip(),
        label=rule.label or llm_spec.label,
        prompts=rule.prompts or llm_spec.prompts,
        color=rule.color or llm_spec.color,
        distance_m=rule.distance_m if rule.distance_m is not None else llm_spec.distance_m,
        tolerance_m=rule.tolerance_m if rule.tolerance_m is not None else llm_spec.tolerance_m,
        body_bearing=rule.body_bearing if rule.body_bearing is not None else llm_spec.body_bearing,
        world_heading=rule.world_heading if rule.world_heading is not None else llm_spec.world_heading,
        report_to=rule.report_to or llm_spec.report_to,
    )


class QwenParser:
    def __init__(self, model_path: str, device: str = "cpu", max_new_tokens: int = 160) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self._tok = None
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = str(Path(self.model_path).expanduser())
        self._tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()
        log.info("qwen parser loaded %s on %s", path, self.device)

    def try_parse(self, text: str) -> Optional[MissionSpec]:
        try:
            self.load()
        except Exception as exc:
            log.warning("qwen parser load failed: %s", exc)
            return None
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text.strip()},
        ]
        prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self._tok(prompt, return_tensors="pt")
        ids = {k: v.to(self.device) for k, v in ids.items()}
        import torch
        try:
            with torch.no_grad():
                out = self._model.generate(
                    **ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                    temperature=1.0, top_p=1.0, top_k=0,
                    pad_token_id=self._tok.eos_token_id,
                )
        except Exception as exc:
            log.warning("qwen parser generate failed: %s", exc)
            return None
        gen = out[0, ids["input_ids"].shape[1]:]
        raw = self._tok.decode(gen, skip_special_tokens=True)
        obj = _extract_json(raw)
        if obj is None:
            log.info("qwen parser no json: %s", raw[:200])
            return None
        return spec_from_llm_json(text, obj)

    def __call__(self, text: str, *, default_tolerance: float = 0.25) -> MissionSpec:
        return merge_rule_llm(text, self.try_parse(text), default_tolerance=default_tolerance)


def make_parse_fn(cfg: Optional[Dict[str, Any]], base: Path) -> Callable[[str], MissionSpec]:
    cfg = dict(cfg or {})
    kind = str(cfg.get("type") or "rules")
    if kind in ("rules", "keyword", ""):
        return parse_instruction
    if kind not in ("qwen", "qwen05", "llm"):
        raise ValueError(f"unknown parser type {kind}")
    model = cfg.get("model") or "weights/parser/Qwen2.5-0.5B-Instruct"
    q = Path(str(model)).expanduser()
    path = str(q if q.is_absolute() else (base / q))
    parser = QwenParser(path, device=str(cfg.get("device", "cpu")),
                        max_new_tokens=int(cfg.get("max_new_tokens", 160)))
    return parser
