"""Deterministic fake probe for dry runs and tests.

It cheats by reading the gold action from `req.meta["gold"]` and returning it with a
configurable probability, so the pipeline, matcher and analysis can be exercised
end-to-end without a model. Accuracy knobs:

  blind_accuracy     P(correct) when meta["mode"] == "blind", decays with horizon
  observed_accuracy  P(correct) when meta["mode"] in ("direct", "cot")
  cot_bonus          added to observed_accuracy when mode == "cot" and the gold
                     action is in `cot_helps_types` (to fake a bimodal Δ)
"""
from __future__ import annotations

import json
import random
from typing import Optional

from ..data.schema import Action
from .base import ProbeModel, ProbeRequest

_WRONG = [
    Action("scroll", direction="down"),
    Action("navigate_back"),
    Action("click", 10, 10),
    Action("type", text="xyz"),
]


class MockProbe(ProbeModel):
    name = "mock"

    def __init__(self, cfg: dict, cache_dir: Optional[str] = None, image_max_side: int = 1024):
        super().__init__(None, image_max_side)
        self.rng = random.Random(cfg.get("seed", 0))
        self.blind_acc = cfg.get("blind_accuracy", 0.6)
        self.obs_acc = cfg.get("observed_accuracy", 0.85)
        self.cot_bonus = cfg.get("cot_bonus", 0.1)
        self.cot_helps_types = set(cfg.get("cot_helps_types", ["complete", "scroll"]))

    def _call(self, req: ProbeRequest) -> tuple[str, Optional[int]]:
        gold: Action = req.meta["gold"]
        mode = req.meta.get("mode", "direct")
        horizon = req.meta.get("horizon", 1)
        if mode == "blind":
            p = self.blind_acc ** horizon
        else:
            p = self.obs_acc
            if mode == "cot" and gold.type in self.cot_helps_types:
                p = min(1.0, p + self.cot_bonus)
            elif mode == "cot":
                p = max(0.0, p - self.cot_bonus / 2)
        act = gold if self.rng.random() < p else self.rng.choice(_WRONG)
        d = {k: v for k, v in act.__dict__.items() if v is not None and k != "raw"}
        txt = json.dumps(d)
        if mode == "cot":
            txt = "Thought: mock reasoning.\nAction: " + txt
        return txt, len(txt.split())
