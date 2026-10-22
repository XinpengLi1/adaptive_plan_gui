"""End-to-end smoke test: synthetic data + mock probe through all stages."""
import json
import sys
from argparse import Namespace
from pathlib import Path

from adaptive_plan.data.schema import load_jsonl, save_jsonl
from adaptive_plan.data.synthetic import generate
from adaptive_plan.experiments import analyze, blind_predict, cot_delta, ngram_stats


def test_pipeline(tmp_path: Path):
    data = tmp_path / "syn.jsonl"
    save_jsonl(generate(20, tmp_path / "screens", seed=1), data)
    assert len(list(load_jsonl(data))) == 20

    s0 = ngram_stats.run(str(data), str(tmp_path / "s0"), None, 10, 2, 0)
    assert s0["n_traj"] == 20
    assert (tmp_path / "s0" / "sample_ids.json").exists()

    blind_predict.run(
        Namespace(data=str(data), out=str(tmp_path / "s1"), probe="mock", model=None, config=None,
                  max_trajs=None, sample_ids=str(tmp_path / "s0" / "sample_ids.json"), max_k=3, strict=False)
    )
    recs = [json.loads(l) for l in open(tmp_path / "s1" / "steps.jsonl")]
    assert recs and all("k_raw" in r for r in recs)

    cot_delta.run(
        Namespace(data=str(data), out=str(tmp_path / "s2"), probe="mock", model=None, config=None,
                  max_trajs=None, sample_ids=str(tmp_path / "s0" / "sample_ids.json"), n_samples=2,
                  temperature=0.7, strict=False)
    )
    recs2 = [json.loads(l) for l in open(tmp_path / "s2" / "steps.jsonl")]
    assert recs2 and all("delta" in r for r in recs2)

    s = analyze.run(
        Namespace(stage1=str(tmp_path / "s1"), stage2=str(tmp_path / "s2"), out=str(tmp_path / "fig"),
                  delta_pos=None, c_large=1.0, c_small=0.1)
    )
    assert "quadrants" in s and "cost_model" in s
    for f in ["k_hist.png", "delta_hist.png", "quadrants.png", "summary.json"]:
        assert (tmp_path / "fig" / f).exists()


def test_resume_skips_done(tmp_path: Path):
    data = tmp_path / "syn.jsonl"
    save_jsonl(generate(3, tmp_path / "screens"), data)
    ns = Namespace(data=str(data), out=str(tmp_path / "s1"), probe="mock", model=None, config=None,
                   max_trajs=None, sample_ids=None, max_k=2, strict=False)
    blind_predict.run(ns)
    n1 = sum(1 for _ in open(tmp_path / "s1" / "steps.jsonl"))
    blind_predict.run(ns)
    n2 = sum(1 for _ in open(tmp_path / "s1" / "steps.jsonl"))
    assert n1 == n2
