# adaptive_plan_gui

Motivation experiments for **adaptive planning in GUI agents**: which steps of a GUI
trajectory can be *packed* (predicted several steps ahead without a new screenshot) and
which steps need *deliberation* (chain-of-thought) before acting.

This repo contains only the offline analysis pipeline that produces the motivation
figures. The chunk policy / executor is a later stage and is not here yet.

---

## 1. The idea in one paragraph

Current GUI agents run a fixed loop: *screenshot → reason → one action → screenshot → …*.
Every step pays for one large-model call and one chain-of-thought, regardless of whether
the step needs either. But real trajectories contain long runs of routine actions
(`click search box → type → press enter`, `open menu → open submenu → toggle`) whose
later steps are fully determined by the first screenshot. We want an agent that
(a) plans several steps at once when the upcoming steps are predictable,
(b) hands those steps to a cheap executor (a grounding model) and only returns to the
large model at the boundary, and
(c) skips chain-of-thought on steps where it does not help.
Before building that, we need to know **where in real trajectories these opportunities
are and how large they are**. That is what this repo measures.

## 2. Two axes, one 2×2

"Needs reasoning" conflates two different things. We measure them separately.

| | needs **no new observation** | needs **new observation** |
|---|---|---|
| needs **no deliberation** | open-loop chunk; small executor | look-then-act; grounding model, no CoT |
| needs **deliberation** | planning point at task start (rare) | large model + CoT |

* **Observation axis** – can the action at step *t+j* be predicted from the screenshot
  at step *t* alone? The largest such *j* is the **packable length** `k_t`.
* **Deliberation axis** – given the screenshot at step *t+1*, does adding a CoT before
  the action change accuracy? The difference is `Δ_t = acc_cot − acc_direct`.

Both are properties of the *(trajectory, model)* pair, not of the trajectory alone, so
both are measured with a probe model rather than read off the data.

## 3. The three stages

| Stage | Script | What it answers | Needs a model? |
|---|---|---|---|
| 0 | `ngram_stats` | Which action n-grams are over-represented? Per-app length distribution? Obvious fixed patterns? | no |
| 1 | `blind_predict` | Distribution of `k_t`; break-point rate per action type; which patterns are packable | yes |
| 2 | `cot_delta` | Distribution of `Δ_t`; is it bimodal; which action types gain from CoT | yes |
| – | `analyze` | Aggregates 1+2 into the 2×2, plots, tables | no |

Stage 0 is cheap and runs first; it sets expectations and chooses a stratified sample
for Stage 1. Stage 1 and 2 share the same probe calls where possible.

### Stage 1 protocol (observation axis)

For each step *t* in a trajectory:

1. **Blind prediction**: input = task goal + action history as text + screenshot at *t*.
   Predict action *t+1*, then *t+2*, … feeding back only the predicted actions, never a
   new screenshot. Stop at the first mismatch. Length before mismatch = `k_t`.
2. **Observed prediction**: input additionally includes the screenshot at *t+1*.
   Predict action *t+1*.
3. If observed prediction is also wrong, the step is marked `hard` and excluded from
   `k_t` statistics (model incapacity, not missing observation).

### Stage 2 protocol (deliberation axis)

For each step *t*, with the screenshot at *t+1* visible:

1. **Direct**: predict the action with no reasoning text.
2. **CoT**: predict the action after a free-form reasoning block.
3. Run both `n_samples` times at temperature > 0; `Δ_t` is the difference in match rate.

### Action matching

An action matches the ground truth if the type matches and, for clicks/long-presses, the
predicted point lies inside the ground-truth element bbox (or within a radius if no bbox);
for typing, the text matches after normalization; for scrolls, the direction matches.
Coordinates are never required to be exact. See `src/adaptive_plan/match.py`.

Human trajectories are not the only valid path, so a strict match under-counts. The
default is the lenient type-level match; a stricter bbox match is available with
`--strict`.

## 4. Data

The pipeline reads a canonical JSONL format (one trajectory per line, see
`src/adaptive_plan/data/schema.py`). Converters:

* `python -m adaptive_plan.data.androidcontrol --tfrecord_dir ... --out data/androidcontrol.jsonl`

AndroidControl is the primary target (per-step screenshots, bboxes, low-level
instructions). GUI-Odyssey and agent-generated OSWorld trajectories can be converted to
the same schema.

## 5. Quick start

```bash
pip install -e .
# Stage 0
python -m adaptive_plan.experiments.ngram_stats --data data/androidcontrol.jsonl --out results/stage0
# Stage 1 (mock probe for a dry run; use --probe openai or --probe qwen for real)
python -m adaptive_plan.experiments.blind_predict --data data/androidcontrol.jsonl \
    --probe mock --max_trajs 200 --max_k 5 --out results/stage1
# Stage 2
python -m adaptive_plan.experiments.cot_delta --data data/androidcontrol.jsonl \
    --probe mock --max_trajs 200 --n_samples 3 --out results/stage2
# Aggregate
python -m adaptive_plan.experiments.analyze --stage1 results/stage1 --stage2 results/stage2 --out results/figures
```

`configs/default.yaml` holds the probe settings (model name, endpoint, temperature).

## 6. What we expect to see (hypotheses)

* `k_t` distribution: most steps `k=1`, a minority `k=2–3`, few `k≥4`. If fewer than
  ~20% of steps have `k≥2`, generic chunking has little room and we should restrict to a
  few high-frequency patterns.
* Break points concentrate on actions that *consume* new information: tapping a search
  result, picking a list item, scrolling an unknown page.
* `Δ_t` is bimodal: near zero or negative on single-target clicks and text entry,
  clearly positive on multi-candidate choices, text comparison, and completion judgment.
* The set of `Δ_t>0` steps overlaps substantially across probe models (property of the
  step), while `k_t` depends more on the probe's app prior (property of the model).

## 7. Related work the motivation leans on

* Uniform planning/reasoning hurts: Learning When to Plan (Goldilocks frequency),
  Focus (all-slow < adaptive), GUI-G1 (longer CoT worse for grounding), AdaGUI-R1
  (−40% reasoning tokens, +5% accuracy).
* Reasoning helps a *different* task subset than it hurts: "Does CoT help mobile GUI
  agents?" (T→F and F→T both non-zero).
* Chunk boundaries must be supervised, not left to RL: SPACE (naive variable-length RL
  collapses to 1 step or over-commits).
* Small grounding specialists beat large generalists at grounding: Agent S2 ablation.
* Plan fidelity drops past 6–7 steps: Planning-as-Routing.

## 8. Layout

```
src/adaptive_plan/
  data/        schema, converters
  probes/      probe model interface + OpenAI-compatible, Qwen2.5-VL, mock
  match.py     action matching
  experiments/ ngram_stats, blind_predict, cot_delta, analyze
configs/       probe + experiment settings
tests/         unit tests (matcher, schema, mock pipeline)
```
