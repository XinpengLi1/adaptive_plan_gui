"""Stage 2: deliberation axis — Δ_t = acc_cot − acc_direct per step.

For each step t (0 <= t < T), with screenshot[t] visible and history[:t]:

  direct:  predict action[t] with no reasoning           (n_samples at temperature τ)
  cot:     predict action[t] after a Thought block       (n_samples at temperature τ)

Δ_t = mean(match_cot) − mean(match_direct). Also records mean CoT length in tokens
so the token cost of deliberation can be reported alongside its benefit.

Output: <out>/steps.jsonl, one record per (traj, t):
  traj_id, t, app, gold_type, acc_direct, acc_cot, delta, cot_tokens, n_samples

Resumable like Stage 1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ..data.schema import Trajectory
from ..match import match_actions
from ..probes import ProbeModel, ProbeRequest
from ..probes.prompts import SYSTEM, observed_cot_prompt, observed_direct_prompt
from .common import append_jsonl, done_keys, history_text, load_cfg, make_probe, select_trajs


def probe_step(
    probe: ProbeModel, traj: Trajectory, t: int, n_samples: int, temperature: float, window: int, strict: bool, max_tokens: int
) -> dict:
    gold = traj.steps[t].action
    sw, sh = traj.screen_w, traj.screen_h
    hist = history_text(traj, t, window)
    img = [traj.steps[t].screenshot]

    def _acc(prompt: str, mode: str) -> tuple[float, float, list]:
        ok = 0
        toks = []
        preds = []
        for i in range(n_samples):
            r = probe(
                ProbeRequest(SYSTEM, prompt, img, temperature, max_tokens, meta={"gold": gold, "mode": mode, "horizon": 1}),
                sample_idx=i,
            )
            m = r.action is not None and match_actions(r.action, gold, strict, sw, sh).matched
            ok += int(m)
            if r.n_tokens_out is not None:
                toks.append(r.n_tokens_out)
            preds.append({"matched": bool(m), "type": r.action.type if r.action else None})
        return ok / n_samples, (sum(toks) / len(toks) if toks else float("nan")), preds

    acc_d, tok_d, preds_d = _acc(observed_direct_prompt(traj.goal, hist), "direct")
    acc_c, tok_c, preds_c = _acc(observed_cot_prompt(traj.goal, hist), "cot")
    return {
        "traj_id": traj.traj_id,
        "t": t,
        "app": traj.app,
        "gold_type": gold.type,
        "instruction": traj.steps[t].instruction,
        "acc_direct": acc_d,
        "acc_cot": acc_c,
        "delta": acc_c - acc_d,
        "direct_tokens": tok_d,
        "cot_tokens": tok_c,
        "n_samples": n_samples,
        "preds_direct": preds_d,
        "preds_cot": preds_c,
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
    n_samples = args.n_samples or ex["n_samples"]
    temperature = args.temperature if args.temperature is not None else ex["sample_temperature"]
    strict = args.strict or ex.get("strict_match", False)
    max_tokens = cfg["probes"][args.probe].get("max_tokens", 512)

    total = sum(len(t) for t in trajs)
    pbar = tqdm(total=total, desc="stage2")
    for traj in trajs:
        for t in range(len(traj)):
            key = f"{traj.traj_id}:{t}"
            if key in done:
                pbar.update(1)
                continue
            rec = probe_step(probe, traj, t, n_samples, temperature, ex["history_window"], strict, max_tokens)
            rec["key"] = key
            append_jsonl(steps_path, rec)
            pbar.update(1)
    pbar.close()
    print(f"done. probe calls this run: {probe.n_calls}. records: {steps_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/stage2")
    ap.add_argument("--probe", default="mock", choices=["mock", "openai", "qwen"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max_trajs", type=int, default=None)
    ap.add_argument("--sample_ids", default=None)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--strict", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
