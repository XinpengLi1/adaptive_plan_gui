"""Stage 1: observation axis — packable length k_t per step.

For each trajectory and each step t (0 <= t < T-1):

  observed:  screenshot[t+1] + history[:t+1]  -> predict action[t+1]
  blind:     screenshot[t]   + history[:t+1] + predicted[...] -> predict action[t+1], t+2, ...
             continue while the prediction matches gold and j < max_k; k_t = #matched

A step with observed==wrong is tagged `hard` (model incapacity) and gets k_t = NaN so
it does not count against packability.

Output: <out>/steps.jsonl with one record per (traj, t):
  traj_id, t, app, gold_type, next_type, observed_ok, hard, k, blind_chain (list of
  {horizon, pred, matched, reason}), n_calls

Resumable: existing (traj_id, t) keys are skipped.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from tqdm import tqdm

from ..data.schema import Action, Trajectory
from ..match import match_actions
from ..probes import ProbeModel, ProbeRequest
from ..probes.prompts import SYSTEM, blind_prompt, observed_direct_prompt
from .common import append_jsonl, done_keys, history_text, load_cfg, make_probe, select_trajs


def _act_dict(a: Action | None) -> dict | None:
    if a is None:
        return None
    return {k: v for k, v in a.__dict__.items() if v is not None and k != "raw"}


def probe_step(
    probe: ProbeModel,
    traj: Trajectory,
    t: int,
    max_k: int,
    window: int,
    strict: bool,
    temperature: float,
    max_tokens: int,
) -> dict:
    T = len(traj)
    gold_next = traj.steps[t + 1].action
    sw, sh = traj.screen_w, traj.screen_h

    # --- observed prediction of action[t+1] from screenshot[t+1]
    req = ProbeRequest(
        SYSTEM,
        observed_direct_prompt(traj.goal, history_text(traj, t + 1, window)),
        images=[traj.steps[t + 1].screenshot],
        temperature=temperature,
        max_tokens=max_tokens,
        meta={"gold": gold_next, "mode": "direct", "horizon": 1},
    )
    r_obs = probe(req)
    obs_ok = r_obs.action is not None and match_actions(r_obs.action, gold_next, strict, sw, sh).matched
    n_calls = 1

    # --- blind chain from screenshot[t]
    chain = []
    predicted_txt: list[str] = []
    k = 0
    for j in range(1, max_k + 1):
        if t + j >= T:
            break
        gold = traj.steps[t + j].action
        req = ProbeRequest(
            SYSTEM,
            blind_prompt(traj.goal, history_text(traj, t + 1, window, predicted_txt), j),
            images=[traj.steps[t].screenshot],
            temperature=temperature,
            max_tokens=max_tokens,
            meta={"gold": gold, "mode": "blind", "horizon": j},
        )
        r = probe(req)
        n_calls += 1
        m = match_actions(r.action, gold, strict, sw, sh) if r.action else None
        ok = bool(m and m.matched)
        chain.append(
            {"horizon": j, "gold": _act_dict(gold), "pred": _act_dict(r.action), "matched": ok, "reason": m.reason if m else "unparsable"}
        )
        if not ok:
            break
        k += 1
        predicted_txt.append(r.action.to_text())

    hard = not obs_ok
    return {
        "traj_id": traj.traj_id,
        "t": t,
        "app": traj.app,
        "gold_type": traj.steps[t].action.type,
        "next_type": gold_next.type,
        "observed_ok": obs_ok,
        "observed_pred": _act_dict(r_obs.action),
        "hard": hard,
        "k": (float("nan") if hard else k),
        "k_raw": k,
        "blind_chain": chain,
        "n_calls": n_calls,
        "remaining": T - 1 - t,
    }


def run(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.config)
    ex = cfg["experiment"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps_path = out / "steps.jsonl"
    probe = make_probe(args.probe, cfg, args.model, str(out / "cache"))
    trajs = select_trajs(args.data, args.max_trajs, args.sample_ids)
    done = done_keys(steps_path, "key")
    max_k = args.max_k or ex["max_k"]
    strict = args.strict or ex.get("strict_match", False)

    total = sum(max(len(t) - 1, 0) for t in trajs)
    pbar = tqdm(total=total, desc="stage1")
    for traj in trajs:
        for t in range(len(traj) - 1):
            key = f"{traj.traj_id}:{t}"
            if key in done:
                pbar.update(1)
                continue
            rec = probe_step(
                probe, traj, t, max_k, ex["history_window"], strict,
                cfg["probes"][args.probe].get("temperature", 0.0),
                cfg["probes"][args.probe].get("max_tokens", 512),
            )
            rec["key"] = key
            rec["k"] = None if (isinstance(rec["k"], float) and math.isnan(rec["k"])) else rec["k"]
            append_jsonl(steps_path, rec)
            pbar.update(1)
    pbar.close()
    print(f"done. probe calls this run: {probe.n_calls}. records: {steps_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/stage1")
    ap.add_argument("--probe", default="mock", choices=["mock", "openai", "qwen"])
    ap.add_argument("--model", default=None, help="override model name in config")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max_trajs", type=int, default=None)
    ap.add_argument("--sample_ids", default=None, help="Stage-0 sample_ids.json")
    ap.add_argument("--max_k", type=int, default=None)
    ap.add_argument("--strict", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
