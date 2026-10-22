"""Action matching: does a predicted action agree with the ground-truth action?

Two levels:

* lenient (default): action type must match; for `type` the normalised text must
  match; for `scroll` the direction must match. Clicks match on type alone.
* strict: additionally, a click/long_press point must fall inside the ground-truth
  bbox, or within `radius_frac` of the screen diagonal if no bbox is available.

Human trajectories are not the unique correct path, so strict matching under-counts
packable steps. Use lenient for the headline numbers and strict as a sensitivity check.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from .data.schema import Action

_EQUIV_TYPES = {
    # treat these pairs as the same type for lenient matching
    frozenset({"click", "long_press"}): "press",
    frozenset({"complete", "wait"}): None,  # never equivalent, listed for clarity
}


def _norm_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("\"'.,!?")
    return s


def _norm_direction(d: Optional[str]) -> str:
    return (d or "").strip().lower()


def _canonical_type(t: str) -> str:
    if t in ("click", "long_press"):
        return "press"
    return t


@dataclass
class MatchResult:
    matched: bool
    reason: str
    type_matched: bool
    content_matched: Optional[bool] = None  # text/direction/point, None if n/a


def match_actions(
    pred: Action,
    gold: Action,
    strict: bool = False,
    screen_w: Optional[int] = None,
    screen_h: Optional[int] = None,
    radius_frac: float = 0.05,
) -> MatchResult:
    """Return whether `pred` matches `gold`.

    radius_frac: when strict and no bbox is available, accept a click within this
    fraction of the screen diagonal (default 5%, ~ AndroidControl's own criterion
    uses element bbox; AITW used 14% of screen distance).
    """
    pt, gt = _canonical_type(pred.type), _canonical_type(gold.type)
    if pt != gt:
        return MatchResult(False, f"type {pred.type}!={gold.type}", False)

    if gt == "type":
        ok = _norm_text(pred.text) == _norm_text(gold.text)
        return MatchResult(ok, "text" if ok else f"text {pred.text!r}!={gold.text!r}", True, ok)

    if gt == "scroll":
        ok = _norm_direction(pred.direction) == _norm_direction(gold.direction)
        return MatchResult(ok, "direction" if ok else "direction mismatch", True, ok)

    if gt == "open_app":
        ok = _norm_text(pred.app) == _norm_text(gold.app) or not gold.app
        return MatchResult(ok, "app" if ok else "app mismatch", True, ok)

    if gt == "press":
        if not strict:
            return MatchResult(True, "type (lenient)", True, None)
        if pred.x is None or pred.y is None:
            return MatchResult(False, "no point in prediction", True, False)
        if gold.bbox:
            x0, y0, x1, y1 = gold.bbox
            ok = x0 <= pred.x <= x1 and y0 <= pred.y <= y1
            return MatchResult(ok, "in bbox" if ok else "outside bbox", True, ok)
        if gold.x is None or gold.y is None:
            return MatchResult(True, "no gold point (lenient fallback)", True, None)
        if screen_w and screen_h:
            diag = math.hypot(screen_w, screen_h)
        else:
            diag = math.hypot(1080, 2400)
        dist = math.hypot(pred.x - gold.x, pred.y - gold.y)
        ok = dist <= radius_frac * diag
        return MatchResult(ok, f"dist={dist:.0f}", True, ok)

    # navigate_back / navigate_home / wait / complete / infeasible / other
    return MatchResult(True, "type", True, None)
