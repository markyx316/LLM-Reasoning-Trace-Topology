# Reasoning Trace Topology as Calibration-Free Uncertainty Quantification

**Core insight:** A single reasoning trace from a reasoning LLM already encodes structural signals about whether the model is likely to be wrong — if we know how to read it.

This project extracts topological features from LLM reasoning traces (backtracking patterns, verification frequency, transition entropy, etc.) and uses them to predict answer correctness, providing a **single-generation, text-surface, black-box** uncertainty quantification method.

## Quick Start

```bash
# 1. Parse traces into behavior episodes
PYTHONPATH=. python src/parsing/rule_based_parser.py \
    --input data/traces/math500_qwen7b_traces.jsonl \
    --output data/parsed/math500_qwen7b_parsed.jsonl

# 2. Extract features
PYTHONPATH=. python src/features/feature_extractor.py \
    --traces data/traces/math500_qwen7b_traces.jsonl \
    --output data/features/math500_qwen7b_features.csv

# 3. Run baselines
PYTHONPATH=. python src/baselines/baseline_a_length_only.py --all
PYTHONPATH=. python src/baselines/baseline_b_lexical.py --all
PYTHONPATH=. python src/baselines/baseline_c_handcrafted.py --all
PYTHONPATH=. python src/baselines/baseline_d_text_encoder.py --all
```

## Architecture

```
src/
├── generation/
│   ├── scoring.py              # Math equivalence, MC matching, GSM8K extraction
│   ├── dataset_loader.py       # Load MATH500, GSM8K, GPQA, ARC-Challenge
│   └── generate_traces.py      # Trace generation (HF Transformers + vLLM)
├── parsing/
│   ├── rule_based_parser.py    # 6-class rule-based behavior parser (NEW)
│   ├── taxonomy.py             # Legacy: 7 cognitive behavior types, 50+ patterns
│   ├── sentence_segmenter.py   # Math-aware sentence boundary detection
│   ├── behavior_classifier.py  # Legacy: full trace → cognitive episode pipeline
│   └── parser_evaluation.py    # Annotation templates + Cohen's Kappa
├── features/
│   ├── feature_extractor.py    # 25 features across 3 groups (NEW)
│   └── feature_pipeline.py     # Legacy: 23-feature pipeline
├── baselines/
│   ├── baseline_a_length_only.py   # Baseline A: trace length + answer length
│   ├── baseline_b_lexical.py       # Baseline B: 7 lexical surface cues
│   ├── baseline_c_handcrafted.py   # Baseline C: all 23 handcrafted features
│   └── baseline_d_text_encoder.py  # Baseline D: TF-IDF text encoder
└── modeling/
    └── train_and_evaluate.py   # CV training, metrics, ablation, transfer
```

## Behavior Taxonomy (6 classes)

The rule-based parser classifies each sentence in a reasoning trace into one of six cognitive behavior types:

| Code | Type | Description | Example signal |
|------|------|-------------|----------------|
| **F** | Forward | Declarative reasoning, computation | "So the answer is...", "Therefore..." |
| **V** | Verify | Checking, confirming, double-checking | "Let me verify...", "Checking: ...", "✓" |
| **X** | Revise | Correction, error acknowledgment | "Wait, that's wrong", "I made an error" |
| **R** | Restart | Full restart, fresh attempt | "Let me start over", "Starting fresh" |
| **H** | Hesitate | Uncertainty, hedging | "Hmm...", "I'm not sure", "Wait..." |
| **C** | Conclude | Final answer delivery | "The answer is", "\\boxed{}" |

Priority order (when multiple patterns match): X > R > V > C > H > F

## Feature Groups (25 features)

| Group | Count | Features |
|-------|-------|----------|
| **Length & Proportion** | 8 | `total_tokens`, `total_episodes`, `prop_forward`, `prop_verify`, `prop_revise`, `prop_restart`, `prop_hesitate`, `prop_conclude` |
| **Structural / Topological** | 10 | `revise_count`, `verify_count`, `restart_count`, `vf_ratio`, `revise_position_mean`, `first_conclude_pos`, `v_clustering`, `max_forward_run`, `transition_entropy`, `cycle_count` |
| **Content-Free Meta** | 7 | `wait_density`, `question_mark_density`, `negation_density`, `repetition_rate_4gram`, `maybe_density`, `verify_density`, `actually_density` |

Key: all meta features are normalized per token (densities, not raw counts). `transition_entropy` is a frequency-weighted Shannon entropy of the behavior bigram transition matrix. `cycle_count` uses Jaccard word-overlap (≥ 0.30) to detect semantically repeated episodes.

## Baselines

| Baseline | Features | Classifier | AUROC range |
|----------|----------|------------|-------------|
| **A — Length Only** | 2: `trace_token_count`, `answer_length` | LR | 0.52 – 0.70 |
| **B — Lexical Cues** | 7: wait, maybe, verify, actually, negation, `?`, repetition | LR | 0.52 – 0.69 |
| **C — Handcrafted** | 23: full feature CSV | LR + RF + XGBoost | 0.54 – 0.78 |
| **D — TF-IDF Encoder** | ~20k unigram+bigram TF-IDF | LR | 0.60 – 0.79 |
| **Ours (structural)** | 25 from 6-class parser | LR / RF / XGBoost | TBD |

Note: Baseline D's advantage over A–C comes from domain vocabulary correlation (content leakage), not structural signals — making it a content-based upper bound, not a fair structural comparison.

All baselines use 5-fold stratified CV. Metrics: AUROC, AUPRC, ECE, Acc@80, Acc@90, AU-Acc-Cov.

## Datasets & Models

| Dataset | Model | Split | Task type |
|---------|-------|-------|-----------|
| MATH500 | Qwen2.5-7B-Instruct | test | Math (open-ended) |
| MATH500 | Llama-3.1-8B-Instruct | test | Math (open-ended) |
| GSM8K | Qwen2.5-7B-Instruct | test | Math (open-ended) |
| GSM8K | Llama-3.1-8B-Instruct | test | Math (open-ended) |
| GPQA Diamond | Qwen2.5-7B-Instruct | test | Science MCQ |
| GPQA Diamond | Llama-3.1-8B-Instruct | test | Science MCQ |
| ARC-Challenge | Qwen2.5-7B-Instruct | test | Science MCQ |
| ARC-Challenge | Llama-3.1-8B-Instruct | test | Science MCQ |

## Test Suite

Each module contains a self-test suite runnable directly:

```bash
PYTHONPATH=. python src/generation/scoring.py              # 27 tests
PYTHONPATH=. python src/parsing/sentence_segmenter.py      # 10 tests
PYTHONPATH=. python src/parsing/rule_based_parser.py       # 30 unit + 1 integration
PYTHONPATH=. python src/features/feature_extractor.py      # 40 tests
PYTHONPATH=. python src/modeling/train_and_evaluate.py     # 18 tests
```

## Compute Requirements

| Task | GPU Hours | Notes |
|------|-----------|-------|
| Primary traces (3,200 items) | 8–12 hrs | 1× A100 or 2× A6000 |
| Feature extraction + training | < 1 hr | CPU only |
| All 4 baselines (--all) | < 30 min | CPU only |

## License

Research code. Model weights are under their respective licenses (Qwen: Apache 2.0, Llama 3.1: Meta Community License).
