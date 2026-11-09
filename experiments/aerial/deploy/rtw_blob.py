"""Raw tensor blob for deployed weights. Magic RTW1, then a JSON header, then bytes."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

MAGIC = b"RTW1"


def is_blob(path: Path | str) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == MAGIC


def pack(path: Path | str, payload: Dict[str, Any]) -> None:
    model = payload["model"]
    if not isinstance(model, dict) or not model:
        raise ValueError("payload model must be a non-empty state dict")
    meta = {k: _jsonable(v) for k, v in payload.items() if k != "model"}
    spec = []
    arrays = []
    for name, value in model.items():
        if not torch.is_tensor(value):
            raise TypeError(f"{name} is {type(value).__name__}, not a tensor")
        tensor = value.detach().cpu().contiguous()
        spec.append({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        arrays.append(tensor.numpy())
    header = json.dumps({"meta": meta, "tensors": spec}, separators=(",", ":")).encode("utf-8")
    dest = Path(path)
    tmp = dest.with_suffix(dest.suffix + ".writing")
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for array in arrays:
            f.write(array.tobytes())
    tmp.replace(dest)


def unpack(path: Path | str) -> Optional[Dict[str, Any]]:
    """Return the checkpoint dict, or None when the file is not an RTW1 blob."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            return None
        (n,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(n).decode("utf-8"))
        tensors = {}
        for spec in header["tensors"]:
            dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
            count = int(np.prod(spec["shape"], dtype=np.int64)) if spec["shape"] else 1
            nbytes = count * torch.empty((), dtype=dtype).element_size()
            raw = f.read(nbytes)
            if len(raw) != nbytes:
                raise ValueError(f"{path} ended inside {spec['name']}")
            tensors[spec["name"]] = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(spec["shape"]).clone()
    return {"model": tensors, **header["meta"]}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"header value {type(value).__name__} is not JSON")
