#!/usr/bin/env bash
# Real run on AndroidControl. Set TFRECORD_DIR and SPLIT_FILE; choose PROBE=openai|qwen.
set -euo pipefail
cd "$(dirname "$0")/.."

TFRECORD_DIR="${TFRECORD_DIR:?set TFRECORD_DIR}"
SPLIT_FILE="${SPLIT_FILE:-}"
PROBE="${PROBE:-openai}"
MODEL="${MODEL:-}"
MAX_EP="${MAX_EP:-}"
SAMPLE_N="${SAMPLE_N:-200}"

conv_args=(--tfrecord_dir "$TFRECORD_DIR" --out data/androidcontrol.jsonl --screens_dir data/androidcontrol/screens)
[[ -n "$SPLIT_FILE" ]] && conv_args+=(--split_file "$SPLIT_FILE" --split test)
[[ -n "$MAX_EP" ]] && conv_args+=(--max_episodes "$MAX_EP")
[[ -f data/androidcontrol.jsonl ]] || python -m adaptive_plan.data.androidcontrol "${conv_args[@]}"

python -m adaptive_plan.experiments.ngram_stats --data data/androidcontrol.jsonl --out results/ac/stage0 --sample_n "$SAMPLE_N"

model_arg=()
[[ -n "$MODEL" ]] && model_arg=(--model "$MODEL")

python -m adaptive_plan.experiments.blind_predict --data data/androidcontrol.jsonl --probe "$PROBE" "${model_arg[@]}" \
    --sample_ids results/ac/stage0/sample_ids.json --out "results/ac/stage1_$PROBE"
python -m adaptive_plan.experiments.cot_delta --data data/androidcontrol.jsonl --probe "$PROBE" "${model_arg[@]}" \
    --sample_ids results/ac/stage0/sample_ids.json --out "results/ac/stage2_$PROBE"
python -m adaptive_plan.experiments.analyze --stage1 "results/ac/stage1_$PROBE" --stage2 "results/ac/stage2_$PROBE" \
    --out "results/ac/figures_$PROBE"
