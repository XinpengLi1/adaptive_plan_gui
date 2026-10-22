"""Stage 0: trajectory-level statistics without any model.

Answers three questions:
  1. Which action-type n-grams (n=2,3,4) are over-represented relative to a unigram
     baseline (lift = P(ngram) / prod P(unigram))?
  2. How do trajectory length and action-type mix vary across apps?
  3. Are there obvious fixed patterns (search triple, menu drill-down, scroll runs)?

Also writes a stratified sample of trajectory ids for Stage 1 so the expensive
probe runs cover the main patterns rather than a random slice.

Outputs (in --out):
  ngrams.csv            n, ngram, count, freq, lift
  per_app.csv           app, n_traj, mean_len, median_len, action-type mix
  length_hist.png       trajectory length histogram
  top_ngrams.png        top-20 trigram lift
  sample_ids.json       stratified sample for Stage 1
  summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..data.schema import Trajectory, load_jsonl


def _ngram_rows(trajs: List[Trajectory], max_n: int = 4, min_count: int = 5) -> pd.DataFrame:
    uni = Counter()
    grams: Dict[int, Counter] = {n: Counter() for n in range(2, max_n + 1)}
    total_uni = 0
    totals = {n: 0 for n in grams}
    for t in trajs:
        seq = t.action_types()
        uni.update(seq)
        total_uni += len(seq)
        for n in grams:
            for i in range(len(seq) - n + 1):
                grams[n][tuple(seq[i : i + n])] += 1
                totals[n] += 1
    p_uni = {k: v / max(total_uni, 1) for k, v in uni.items()}
    rows = []
    for a, c in uni.items():
        rows.append({"n": 1, "ngram": a, "count": c, "freq": c / max(total_uni, 1), "lift": 1.0})
    for n, cnt in grams.items():
        for g, c in cnt.items():
            if c < min_count:
                continue
            freq = c / max(totals[n], 1)
            base = math.prod(p_uni[x] for x in g)
            rows.append(
                {"n": n, "ngram": " > ".join(g), "count": c, "freq": freq, "lift": freq / base if base > 0 else float("nan")}
            )
    df = pd.DataFrame(rows).sort_values(["n", "lift"], ascending=[True, False])
    return df


def _per_app(trajs: List[Trajectory]) -> pd.DataFrame:
    by_app: Dict[str, List[Trajectory]] = defaultdict(list)
    for t in trajs:
        by_app[t.app or "unknown"].append(t)
    rows = []
    for app, ts in by_app.items():
        lens = [len(t) for t in ts]
        mix = Counter(a for t in ts for a in t.action_types())
        tot = sum(mix.values()) or 1
        row = {
            "app": app,
            "n_traj": len(ts),
            "mean_len": sum(lens) / len(lens),
            "median_len": sorted(lens)[len(lens) // 2],
            "max_len": max(lens),
        }
        for k, v in mix.items():
            row[f"frac_{k}"] = v / tot
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_traj", ascending=False)


# Hand-named patterns we expect to be packable. Each is a regex over the
# space-joined action-type sequence. Reported as counts per 1000 steps.
NAMED_PATTERNS = {
    "search_triple": r"click type (click|complete|navigate_back|scroll)",
    "type_then_press": r"type click",
    "scroll_run_2+": r"scroll scroll",
    "scroll_run_3+": r"scroll scroll scroll",
    "click_run_3+": r"click click click",
    "back_then_click": r"navigate_back click",
    "open_app_then_click": r"open_app click",
}


def _named_pattern_counts(trajs: List[Trajectory]) -> Dict[str, float]:
    import re

    total_steps = sum(len(t) for t in trajs) or 1
    out = {}
    for name, pat in NAMED_PATTERNS.items():
        rx = re.compile(pat)
        c = 0
        for t in trajs:
            s = " ".join(t.action_types())
            c += len(rx.findall(s))
        out[name] = 1000.0 * c / total_steps
    return out


def _stratified_sample(trajs: List[Trajectory], n: int, seed: int = 0) -> List[str]:
    """Sample trajectories stratified by (app, length bucket)."""
    rng = random.Random(seed)
    buckets: Dict[tuple, List[Trajectory]] = defaultdict(list)
    for t in trajs:
        lb = "short" if len(t) <= 4 else "mid" if len(t) <= 9 else "long"
        buckets[(t.app or "unknown", lb)].append(t)
    keys = list(buckets)
    rng.shuffle(keys)
    out: List[str] = []
    while len(out) < n and any(buckets.values()):
        for k in keys:
            if buckets[k]:
                out.append(buckets[k].pop(rng.randrange(len(buckets[k]))).traj_id)
                if len(out) >= n:
                    break
    return out


def _plots(trajs: List[Trajectory], ngrams: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lens = [len(t) for t in trajs]
    plt.figure(figsize=(6, 3.5))
    plt.hist(lens, bins=range(1, max(lens) + 2), color="#4C72B0", edgecolor="white")
    plt.xlabel("trajectory length (steps)")
    plt.ylabel("count")
    plt.title("Trajectory length distribution")
    plt.tight_layout()
    plt.savefig(out / "length_hist.png", dpi=150)
    plt.close()

    tri = ngrams[ngrams.n == 3].head(20)
    if len(tri):
        plt.figure(figsize=(7, 5))
        plt.barh(tri.ngram[::-1], tri.lift[::-1], color="#55A868")
        plt.xlabel("lift over unigram baseline")
        plt.title("Top trigrams by lift")
        plt.tight_layout()
        plt.savefig(out / "top_ngrams.png", dpi=150)
        plt.close()


def run(data: str, out: str, max_trajs: int | None, sample_n: int, min_count: int, seed: int) -> dict:
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    trajs = list(load_jsonl(data, max_trajs))
    if not trajs:
        raise SystemExit("no trajectories loaded")

    ngrams = _ngram_rows(trajs, min_count=min_count)
    ngrams.to_csv(out_p / "ngrams.csv", index=False)
    per_app = _per_app(trajs)
    per_app.to_csv(out_p / "per_app.csv", index=False)
    named = _named_pattern_counts(trajs)
    sample = _stratified_sample(trajs, sample_n, seed)
    (out_p / "sample_ids.json").write_text(json.dumps(sample, indent=1))
    _plots(trajs, ngrams, out_p)

    lens = [len(t) for t in trajs]
    summary = {
        "n_traj": len(trajs),
        "n_steps": sum(lens),
        "mean_len": sum(lens) / len(lens),
        "median_len": sorted(lens)[len(lens) // 2],
        "n_apps": int(per_app.shape[0]),
        "named_patterns_per_1000_steps": named,
        "top_bigrams_by_lift": ngrams[ngrams.n == 2].head(10)[["ngram", "count", "lift"]].to_dict("records"),
        "top_trigrams_by_lift": ngrams[ngrams.n == 3].head(10)[["ngram", "count", "lift"]].to_dict("records"),
        "sample_n": len(sample),
    }
    (out_p / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/stage0")
    ap.add_argument("--max_trajs", type=int, default=None)
    ap.add_argument("--sample_n", type=int, default=200, help="stratified sample size for Stage 1")
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    s = run(args.data, args.out, args.max_trajs, args.sample_n, args.min_count, args.seed)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
