"""Shared helpers for the probe-based stages."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import yaml

from ..data.schema import Trajectory, load_jsonl
from ..probes import ProbeModel, build_probe

_DEFAULT_CFG = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def load_cfg(path: Optional[str]) -> dict:
    with open(path or _DEFAULT_CFG) as f:
        return yaml.safe_load(f)


def make_probe(kind: str, cfg: dict, model: Optional[str], cache_dir: Optional[str]) -> ProbeModel:
    pcfg = dict(cfg["probes"][kind])
    if model:
        pcfg["model"] = model
    return build_probe(kind, pcfg, cache_dir=cache_dir, image_max_side=cfg["experiment"]["image_max_side"])


def select_trajs(data: str, max_trajs: Optional[int], sample_ids: Optional[str]) -> List[Trajectory]:
    """Load trajectories, optionally restricted to a Stage-0 stratified sample."""
    ids = None
    if sample_ids:
        ids = set(json.loads(Path(sample_ids).read_text()))
    out = []
    for t in load_jsonl(data):
        if ids is not None and t.traj_id not in ids:
            continue
        out.append(t)
        if max_trajs is not None and ids is None and len(out) >= max_trajs:
            break
    if ids is not None and max_trajs is not None:
        out = out[:max_trajs]
    return out


def history_text(traj: Trajectory, upto: int, window: int, predicted: Optional[List[str]] = None) -> List[str]:
    """Action history as text up to (not including) step `upto`, plus any predicted
    actions appended (marked) for blind multi-step rollouts."""
    hist = [s.action.to_text() for s in traj.steps[:upto]]
    hist = hist[-window:]
    if predicted:
        hist = hist + [f"{p}  (predicted, not yet verified)" for p in predicted]
    return hist


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_keys(path: Path, key: str) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)[key] for l in open(path) if l.strip()}
