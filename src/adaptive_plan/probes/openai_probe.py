"""OpenAI-compatible chat-completions probe (OpenAI, vLLM, etc.)."""
from __future__ import annotations

import os
import time
from typing import Optional

from .base import ProbeModel, ProbeRequest, encode_image


class OpenAIProbe(ProbeModel):
    def __init__(self, cfg: dict, cache_dir: Optional[str] = None, image_max_side: int = 1024):
        super().__init__(cache_dir, image_max_side)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("pip install openai") from e
        self.model = cfg.get("model", "gpt-4o")
        self.name = f"openai:{self.model}"
        api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
        self.client = OpenAI(api_key=api_key, base_url=cfg.get("base_url") or None)
        self.default_temperature = cfg.get("temperature", 0.0)
        self.default_max_tokens = cfg.get("max_tokens", 512)

    def _call(self, req: ProbeRequest) -> tuple[str, Optional[int]]:
        content = []
        for p in req.images:
            b64, mime = encode_image(p, self.image_max_side)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        content.append({"type": "text", "text": req.prompt})
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": req.system}, {"role": "user", "content": content}],
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                )
                txt = r.choices[0].message.content or ""
                n_out = r.usage.completion_tokens if r.usage else None
                return txt, n_out
            except Exception as e:  # rate limits, transient errors
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        return "", None
