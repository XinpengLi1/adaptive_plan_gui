"""Probe model interface + response parsing shared by all backends."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..data.schema import Action


@dataclass
class ProbeRequest:
    system: str
    prompt: str
    images: List[str] = field(default_factory=list)  # file paths, in order
    temperature: float = 0.0
    max_tokens: int = 512
    # metadata for mock / caching; not sent to the model
    meta: dict = field(default_factory=dict)


@dataclass
class ProbeResponse:
    text: str
    action: Optional[Action]
    thought: Optional[str] = None
    n_tokens_out: Optional[int] = None
    cached: bool = False


_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_action(text: str) -> tuple[Optional[Action], Optional[str]]:
    """Extract (action, thought) from model text. Takes the LAST JSON object found."""
    thought = None
    m = re.search(r"Thought:\s*(.*?)(?:\n\s*Action:|$)", text, re.S)
    if m:
        thought = m.group(1).strip()
    objs = _JSON_RE.findall(text)
    for cand in reversed(objs):
        try:
            d = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if "type" in d:
            return Action.from_dict({**d, "raw": d}), thought
    return None, thought


def encode_image(path: str, max_side: int = 1024) -> tuple[str, str]:
    """Return (base64, mime) for an image, downscaled so max(w,h) <= max_side."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


class ProbeModel:
    """Base class. Subclasses implement `_call`. Caching is handled here."""

    name: str = "base"

    def __init__(self, cache_dir: Optional[str] = None, image_max_side: int = 1024):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_max_side = image_max_side
        self.n_calls = 0

    def _cache_key(self, req: ProbeRequest, sample_idx: int) -> str:
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(req.system.encode())
        h.update(req.prompt.encode())
        for p in req.images:
            h.update(p.encode())
            try:
                h.update(str(Path(p).stat().st_size).encode())
            except OSError:
                pass
        h.update(f"{req.temperature}:{req.max_tokens}:{sample_idx}".encode())
        return h.hexdigest()

    def __call__(self, req: ProbeRequest, sample_idx: int = 0) -> ProbeResponse:
        key = self._cache_key(req, sample_idx) if self.cache_dir else None
        if key:
            f = self.cache_dir / f"{key}.json"
            if f.exists():
                d = json.loads(f.read_text())
                act, thought = parse_action(d["text"])
                return ProbeResponse(d["text"], act, thought, d.get("n_tokens_out"), cached=True)
        text, n_out = self._call(req)
        self.n_calls += 1
        if key:
            (self.cache_dir / f"{key}.json").write_text(json.dumps({"text": text, "n_tokens_out": n_out}))
        act, thought = parse_action(text)
        return ProbeResponse(text, act, thought, n_out)

    def _call(self, req: ProbeRequest) -> tuple[str, Optional[int]]:
        raise NotImplementedError


def build_probe(kind: str, cfg: dict, cache_dir: Optional[str] = None, image_max_side: int = 1024) -> ProbeModel:
    if kind == "mock":
        from .mock import MockProbe

        return MockProbe(cfg, cache_dir=None, image_max_side=image_max_side)
    if kind == "openai":
        from .openai_probe import OpenAIProbe

        return OpenAIProbe(cfg, cache_dir=cache_dir, image_max_side=image_max_side)
    if kind == "qwen":
        from .qwen_probe import QwenProbe

        return QwenProbe(cfg, cache_dir=cache_dir, image_max_side=image_max_side)
    raise ValueError(f"unknown probe kind {kind}")
