# Reasoning Trace Topology as Calibration-Free Uncertainty Quantification

**Core insight:** A single reasoning trace from a reasoning LLM already encodes structural signals about whether the model is likely to be wrong — if we know how to read it.

This project extracts topological features from LLM reasoning traces (backtracking patterns, verification frequency, transition entropy, etc.) and uses them to predict answer correctness, providing a **single-generation, text-surface, black-box** uncertainty quantification method.

## Quick Start

```bash
# 1. Validate the pipeline (no GPU needed)
PYTHONPATH=. python scripts/validate_pipeline.py

# 2. Generate traces (requires GPU)
./scripts/run_generation.sh pilot        # 50-item pilot study
./scripts/run_generation.sh full         # All datasets

# 3. Run experiments
./scripts/run_experiments.sh all         # Parse → Features → Train → Evaluate
```

## Architecture

```
src/
├── generation/
│   ├── scoring.py              # Math equivalence, MC matching, GSM8K extraction
│   ├── dataset_loader.py       # Load MATH500, GSM8K, GPQA, ARC-Challenge
│   └── generate_traces.py      # Trace generation (HF Transformers + vLLM)
├── parsing/
│   ├── taxonomy.py             # 7 cognitive behavior types, 50+ detection patterns
│   ├── sentence_segmenter.py   # Math-aware sentence boundary detection
│   ├── behavior_classifier.py  # Full trace → cognitive episode pipeline
│   └── parser_evaluation.py    # Annotation templates + Cohen's Kappa
├── features/
│   └── feature_pipeline.py     # 23 features across 3 groups
├── baselines/
│   └── baselines.py            # 5 UQ baselines (verbalized, SC, SE, PPL, length)
└── modeling/
    └── train_and_evaluate.py   # CV training, metrics, ablation, transfer
```

## Feature Groups (23 features)

| Group | Features | Key Signals |
|-------|----------|-------------|
| **Length & Proportion** (9) | Token count, episode count, behavior proportions | Longer traces with more backtracking → lower confidence |
| **Structural** (10) | Backtrack position, V/F ratio, transition entropy, cycles | Late backtracks + high entropy → model "knew something was wrong" |
| **Content-Free Meta** (4) | Wait-word ratio, question marks, negations, 4-gram repetition | Parser-independent surface statistics |

## Baselines

| Method | Cost | Access | Implementation |
|--------|------|--------|----------------|
| Trace length | 1 gen | Text | `baselines.trace_length_confidence()` |
| Verbalized confidence | 1 gen | Text | `baselines.verbalized_confidence()` |
| Self-consistency (N=8) | 8 gen | Text | `baselines.self_consistency_confidence()` |
| Semantic entropy (N=8) | 8 gen | Text+NLI | `baselines.semantic_entropy_confidence()` |
| Perplexity | 1 gen | White-box | `baselines.perplexity_confidence()` |
| **Ours (structural)** | **1 gen** | **Text** | `baselines.structural_confidence()` |

## Test Suite

Every module has self-tests. Run individually or all at once:

```bash
PYTHONPATH=. python src/generation/scoring.py       # 27 tests
PYTHONPATH=. python src/parsing/taxonomy.py          # 25 tests
PYTHONPATH=. python src/parsing/sentence_segmenter.py # 10 tests
PYTHONPATH=. python src/parsing/behavior_classifier.py # 9 tests
PYTHONPATH=. python src/features/feature_pipeline.py  # 38 tests
PYTHONPATH=. python src/baselines/baselines.py        # 20 tests
PYTHONPATH=. python src/modeling/train_and_evaluate.py # 18 tests

# Full pipeline validation (all of the above + integration tests):
PYTHONPATH=. python scripts/validate_pipeline.py      # 9 steps, 150+ tests
```

## Compute Requirements

| Task | GPU Hours | Notes |
|------|-----------|-------|
| Primary traces (3,200 items) | 8-12 hrs | 1× A100 or 2× A6000 |
| Self-consistency (N=8, 700 items) | 16-22 hrs | For baselines only |
| Feature extraction + training | < 1 hr | CPU only |

## License

Research code. Model weights are under the DeepSeek-R1 MIT license.
