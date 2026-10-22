"""Aggregate Stage 1 and Stage 2 into the motivation tables and figures.

Inputs:
  --stage1 <dir>   containing steps.jsonl from blind_predict
  --stage2 <dir>   containing steps.jsonl from cot_delta
Either can be omitted; the 2x2 needs both.

Outputs in --out:
  k_hist.png                 distribution of packable length k_t (hard steps excluded)
  breakpoint_by_type.csv     per next-action type: P(k=0) i.e. "needs observation"
  breakpoint_by_type.png
  delta_hist.png             distribution of Δ_t
  delta_by_type.csv          per action type: mean Δ, P(Δ>0), P(Δ<0), mean CoT tokens
  delta_by_type.png
  quadrants.csv              2x2 counts and fractions (steps with both measurements)
  quadrants.png
  packable_runs.csv          contiguous runs of k>=1 steps: length distribution
  summary.json               headline numbers
  cost_model.json            projected large-model call savings under the 2x2

Thresholds: a step "needs observation" if k_t == 0; "needs deliberation" if Δ_t >= delta_pos
(default 1/n_samples, i.e. at least one more correct sample with CoT).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load(p: str | None) -> pd.DataFrame | None:
    if not p:
        return None
    f = Path(p) / "steps.jsonl"
    if not f.exists():
        return None
    return pd.read_json(f, lines=True)


def _save_bar(series: pd.Series, title: str, ylabel: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.5, 3.8))
    series.plot.bar(color="#4C72B0")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def analyze_stage1(s1: pd.DataFrame, out: Path, summary: dict) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = s1[~s1.hard].copy()
    summary["stage1"] = {
        "n_steps": int(len(s1)),
        "n_hard": int(s1.hard.sum()),
        "hard_frac": float(s1.hard.mean()),
        "k_mean": float(valid.k.mean()) if len(valid) else None,
        "k_ge1_frac": float((valid.k >= 1).mean()) if len(valid) else None,
        "k_ge2_frac": float((valid.k >= 2).mean()) if len(valid) else None,
        "k_ge3_frac": float((valid.k >= 3).mean()) if len(valid) else None,
        "k_hist": valid.k.value_counts().sort_index().to_dict() if len(valid) else {},
    }

    if len(valid):
        plt.figure(figsize=(5.5, 3.5))
        vc = valid.k.value_counts(normalize=True).sort_index()
        plt.bar(vc.index, vc.values, color="#4C72B0")
        plt.xlabel("packable length k_t (steps predictable from screenshot t)")
        plt.ylabel("fraction of steps")
        plt.title("Observation axis: how far ahead is predictable?")
        plt.tight_layout()
        plt.savefig(out / "k_hist.png", dpi=150)
        plt.close()

    # break-point rate by the type of the NEXT action (the one that would be predicted blind)
    bp = (
        valid.groupby("next_type")
        .agg(n=("k", "size"), needs_obs=("k", lambda x: float((x == 0).mean())), k_mean=("k", "mean"))
        .sort_values("needs_obs", ascending=False)
    )
    bp.to_csv(out / "breakpoint_by_type.csv")
    if len(bp):
        _save_bar(bp.needs_obs, "P(needs new observation) by next action type", "P(k_t = 0)", out / "breakpoint_by_type.png")

    # contiguous runs of k>=1 (i.e. steps whose successor was predictable)
    runs = []
    for tid, g in valid.sort_values("t").groupby("traj_id"):
        cur = 0
        for v in g.k.values:
            if v >= 1:
                cur += 1
            else:
                if cur:
                    runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
    if runs:
        pd.Series(runs, name="run_len").value_counts().sort_index().to_csv(out / "packable_runs.csv")
        summary["stage1"]["packable_run_mean"] = float(np.mean(runs))
        summary["stage1"]["packable_run_max"] = int(max(runs))
    return valid


def analyze_stage2(s2: pd.DataFrame, out: Path, summary: dict, delta_pos: float) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary["stage2"] = {
        "n_steps": int(len(s2)),
        "acc_direct_mean": float(s2.acc_direct.mean()),
        "acc_cot_mean": float(s2.acc_cot.mean()),
        "delta_mean": float(s2.delta.mean()),
        "delta_pos_frac": float((s2.delta >= delta_pos).mean()),
        "delta_neg_frac": float((s2.delta <= -delta_pos).mean()),
        "delta_zero_frac": float((s2.delta.abs() < delta_pos).mean()),
        "cot_tokens_mean": float(s2.cot_tokens.mean()) if s2.cot_tokens.notna().any() else None,
        "direct_tokens_mean": float(s2.direct_tokens.mean()) if s2.direct_tokens.notna().any() else None,
    }
    plt.figure(figsize=(5.5, 3.5))
    plt.hist(s2.delta, bins=np.linspace(-1, 1, 21), color="#DD8452", edgecolor="white")
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("Δ_t = acc_cot − acc_direct")
    plt.ylabel("steps")
    plt.title("Deliberation axis: does CoT help this step?")
    plt.tight_layout()
    plt.savefig(out / "delta_hist.png", dpi=150)
    plt.close()

    bt = (
        s2.groupby("gold_type")
        .agg(
            n=("delta", "size"),
            delta_mean=("delta", "mean"),
            p_pos=("delta", lambda x: float((x >= delta_pos).mean())),
            p_neg=("delta", lambda x: float((x <= -delta_pos).mean())),
            acc_direct=("acc_direct", "mean"),
            acc_cot=("acc_cot", "mean"),
            cot_tokens=("cot_tokens", "mean"),
        )
        .sort_values("delta_mean", ascending=False)
    )
    bt.to_csv(out / "delta_by_type.csv")
    if len(bt):
        _save_bar(bt.delta_mean, "Mean Δ_t by action type", "Δ", out / "delta_by_type.png")
    return s2


def analyze_quadrants(s1: pd.DataFrame, s2: pd.DataFrame, out: Path, summary: dict, delta_pos: float) -> None:
    """Join on the action being predicted: stage1 row (traj, t) predicts action t+1,
    stage2 row (traj, t) evaluates action t. So join s1.t+1 == s2.t."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = s1[["traj_id", "t", "k"]].copy()
    a["t"] = a["t"] + 1
    b = s2[["traj_id", "t", "delta", "gold_type"]]
    j = a.merge(b, on=["traj_id", "t"])
    if j.empty:
        return
    j["needs_obs"] = j.k == 0
    j["needs_delib"] = j.delta >= delta_pos
    q = j.groupby(["needs_delib", "needs_obs"]).size().unstack(fill_value=0)
    q = q.reindex(index=[False, True], columns=[False, True], fill_value=0)
    q.index = ["no deliberation", "deliberation"]
    q.columns = ["no observation", "observation"]
    frac = q / q.values.sum()
    pd.concat({"count": q, "frac": frac}, axis=1).to_csv(out / "quadrants.csv")
    summary["quadrants"] = {
        "n": int(q.values.sum()),
        "open_loop_chunk": float(frac.loc["no deliberation", "no observation"]),
        "look_then_act": float(frac.loc["no deliberation", "observation"]),
        "plan_point": float(frac.loc["deliberation", "no observation"]),
        "full_reasoning": float(frac.loc["deliberation", "observation"]),
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(frac.values, cmap="Blues", vmin=0, vmax=max(0.5, frac.values.max()))
    ax.set_xticks([0, 1]); ax.set_xticklabels(frac.columns)
    ax.set_yticks([0, 1]); ax.set_yticklabels(frac.index)
    for i in range(2):
        for k in range(2):
            ax.text(k, i, f"{frac.values[i,k]:.0%}\n(n={q.values[i,k]})", ha="center", va="center", fontsize=11)
    ax.set_title("Where do steps fall?")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out / "quadrants.png", dpi=150)
    plt.close()

    # per-type quadrant mix
    (j.groupby("gold_type")[["needs_obs", "needs_delib"]].mean().to_csv(out / "quadrant_by_type.csv"))


def cost_model(summary: dict, out: Path, c_large: float = 1.0, c_small: float = 0.1) -> None:
    """Projected large-model calls per step under three policies, from measured fractions.

    baseline:   1 large call per step
    skip-CoT:   1 large call per step but no CoT tokens where Δ≈0 (token saving only)
    chunk+route: large call only at chunk boundaries that need deliberation; other steps
                 go to a small executor. Uses the 2x2 fractions + packable_run_mean.
    """
    q = summary.get("quadrants")
    s1 = summary.get("stage1", {})
    if not q:
        return
    run_len = s1.get("packable_run_mean") or 1.0
    # fraction of steps that are chunk interiors ~ k>=1 steps; each run costs 1 large call
    frac_chunk_interior = s1.get("k_ge1_frac", 0.0)
    large_calls_per_step = (1 - frac_chunk_interior) + frac_chunk_interior / max(run_len, 1.0)
    # further: steps that need no deliberation can be routed to the small model
    routed_small = q["look_then_act"] + q["open_loop_chunk"]
    large_calls_per_step_routed = large_calls_per_step * (1 - routed_small)
    cost = {
        "assumptions": {"c_large": c_large, "c_small": c_small, "packable_run_mean": run_len},
        "baseline_cost_per_step": c_large,
        "chunk_only_large_calls_per_step": large_calls_per_step,
        "chunk_only_cost_per_step": large_calls_per_step * c_large + (1 - large_calls_per_step) * c_small,
        "chunk_and_route_large_calls_per_step": large_calls_per_step_routed,
        "chunk_and_route_cost_per_step": large_calls_per_step_routed * c_large
        + (1 - large_calls_per_step_routed) * c_small,
    }
    cost["saving_chunk_only"] = 1 - cost["chunk_only_cost_per_step"] / c_large
    cost["saving_chunk_and_route"] = 1 - cost["chunk_and_route_cost_per_step"] / c_large
    (out / "cost_model.json").write_text(json.dumps(cost, indent=1))
    summary["cost_model"] = cost


def run(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    s1 = _load(args.stage1)
    s2 = _load(args.stage2)
    if s1 is None and s2 is None:
        raise SystemExit("need at least one of --stage1 / --stage2 with steps.jsonl")
    delta_pos = args.delta_pos
    if s2 is not None and delta_pos is None:
        delta_pos = 1.0 / max(int(s2.n_samples.iloc[0]), 1)
    v1 = analyze_stage1(s1, out, summary) if s1 is not None else None
    if s2 is not None:
        analyze_stage2(s2, out, summary, delta_pos)
    if v1 is not None and s2 is not None:
        analyze_quadrants(v1, s2, out, summary, delta_pos)
        cost_model(summary, out, args.c_large, args.c_small)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", default=None)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--delta_pos", type=float, default=None, help="threshold for 'CoT helps'; default 1/n_samples")
    ap.add_argument("--c_large", type=float, default=1.0)
    ap.add_argument("--c_small", type=float, default=0.1)
    s = run(ap.parse_args())
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
