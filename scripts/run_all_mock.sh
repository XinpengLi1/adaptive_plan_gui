#!/usr/bin/env bash
# End-to-end dry run on synthetic data with the mock probe.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m adaptive_plan.data.synthetic --out data/synthetic.jsonl --n 60
python -m adaptive_plan.experiments.ngram_stats --data data/synthetic.jsonl --out results/mock/stage0 --min_count 2 --sample_n 30
python -m adaptive_plan.experiments.blind_predict --data data/synthetic.jsonl --probe mock \
    --sample_ids results/mock/stage0/sample_ids.json --max_k 4 --out results/mock/stage1
python -m adaptive_plan.experiments.cot_delta --data data/synthetic.jsonl --probe mock \
    --sample_ids results/mock/stage0/sample_ids.json --n_samples 3 --out results/mock/stage2
python -m adaptive_plan.experiments.analyze --stage1 results/mock/stage1 --stage2 results/mock/stage2 --out results/mock/figures
echo "figures in results/mock/figures"
