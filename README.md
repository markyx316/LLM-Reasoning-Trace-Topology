# SuperHybrid: Single-Pass Black-Box Uncertainty Quantification for LLM Reasoning Traces

Calibration-free uncertainty quantification (UQ) for large reasoning models. Given a
**single** reasoning trace and the generator's hidden state at one position,
**SuperHybrid** predicts whether the model's final answer is correct — at one
generation cost, with no logits, no sampling, and no model-side self-report.

📄 **Final report:** [`reports/FINAL_REPORT.pdf`](reports/FINAL_REPORT.pdf)
🔁 **Reproduction walkthrough:** [`REPRODUCING.md`](REPRODUCING.md)

---

## Headline result

Across **6,378 traces** spanning 4 benchmarks (MATH500, GSM8K, GPQA-Diamond,
ARC-Challenge) and 2 reasoning-distilled models (DeepSeek-R1-Distill-Qwen-7B and
Llama-8B), SuperHybrid reaches **pooled AUROC 0.815 (LR meta-learner) /
0.807 (RF) with ECE 0.042**, beating a fine-tuned DeBERTa-v3 read of the
trace by **+0.05 AUROC** (paired DeLong p < 10⁻¹⁹) and halving its calibration
error. Peak per-cell: **0.934** on MATH500-Qwen7B.

| Method                         | AUROC | ECE   |
|--------------------------------|------:|------:|
| Length-only LR                 | 0.595 | —     |
| Plain DeBERTa-v3 (trace tail)  | 0.762 | 0.090 |
| Problem-conditioned DeBERTa    | 0.788 | 0.086 |
| Multi-layer hidden-state probe | 0.776 | —     |
| **SuperHybrid (LR)**           | **0.815** | 0.080 |
| **SuperHybrid (RF)**           | **0.807** | **0.042** |

Full numbers, ablations, and cross-domain transfer in
[`reports/FINAL_REPORT.pdf`](reports/FINAL_REPORT.pdf).

---

## How SuperHybrid works (one paragraph)

SuperHybrid is a **stacking pipeline** over four complementary views of a trace:
(i) a fine-tuned DeBERTa-v3 read of the trace tail, (ii) a problem-conditioned
cross-encoder DeBERTa, (iii) a small MLP probe over the generator's hidden
states at the answer-marker token, concatenated across four decoder layers,
and (iv) a 28-dim behavioral-feature vector parsed from the trace text using
a six-class cognitive behavior taxonomy
(forward / verify / revise / restart / hesitate / conclude). The four base
predictors emit out-of-fold (OOF) probabilities; a meta-learner
(logistic regression / random forest / XGBoost) fits on those OOFs in a fresh
outer 5-fold CV — making the stacked prediction **leakage-safe at the item
level**. Random forest is the recommended terminal classifier because it
halves the calibration error at indistinguishable AUROC.

---

## Repository layout

```
.
├── reports/
│   ├── FINAL_REPORT.pdf         <- compiled paper (4 pages body + 3 appendices)
│   ├── FINAL_REPORT.tex         <- LaTeX source (NeurIPS 2025 template)
│   ├── neurips_2025.sty         <- style file for the template
│   ├── route_ab/                <- per-group DeLong CIs, ablation tables
│   └── month3/                  <- experiment-by-experiment summaries
├── src/
│   ├── analysis/                <- DeLong CI and paired-DeLong machinery
│   ├── baselines/               <- Baselines A-D (length, lexical, handcrafted, TF-IDF)
│   ├── features/                <- Behavioral, recurrence, n-gram, graph, timing,
│   │                               persistent-homology, shapelet feature extractors
│   ├── generation/              <- Trace generation + answer scoring
│   ├── modeling/                <- Probes, Step Transformer, GNN, hybrid stackers
│   └── parsing/                 <- Sentence segmentation + behavior taxonomy
├── scripts/
│   ├── run_*.sh                 <- Convenience drivers (generation, parsing, training)
│   ├── sbatch_*.sh              <- SLURM job scripts for the HPC steps
│   └── phase3/                  <- Phase 3 experiment scripts (probes, OOF stacking)
├── data/
│   ├── traces/                  <- Raw generated traces (one .jsonl per dataset x model)
│   ├── parsed/                  <- Parsed behavior-episode sequences
│   ├── features/                <- All feature CSVs (28-feat behavioral + extended)
│   ├── hybrid_table.parquet     <- Joined feature matrix (6,344 x 366) for tuning
│   └── optuna_hybrid_v1_clean.db <- Optuna study log (XGBoost stacker tuning)
├── results/                     <- Per-experiment OOFs, paired-DeLong tests, summaries
├── README.md                    <- This file
├── REPRODUCING.md               <- Step-by-step walkthrough to reproduce headline numbers
├── PLAN_ROUTE_AB.md             <- Developer notes: design of the Route-A/B feature blocks
├── HPC_WALKTHROUGH_ROUTE_AB.md  <- Developer notes: HPC submission protocol
├── HPC_HYBRID_TUNING.md         <- Developer notes: Optuna tuning protocol
├── RESEARCH_GUIDE.md            <- Developer notes: reproduction tips for the project
└── research_proposal.md         <- Original (pre-research) proposal; preserved for reference
```

The four root `*.md` files other than `README.md` and `REPRODUCING.md` are
**developer notes** preserved for archival; the canonical entry points are
this README and `REPRODUCING.md`.

---

## Install

Tested on Ubuntu 22.04 / WSL2 with Python 3.11.

```bash
# 1. Clone
git clone <this repo URL>
cd LLM-Reasoning-Trace-Topology

# 2. Conda environment (recommended)
conda create -n superhybrid python=3.11
conda activate superhybrid

# 3. Python deps
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. PyTorch (install matching your CUDA; CPU-only is fine for stacking only)
#    Example for CUDA 12.1:
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Set the repo root on PYTHONPATH (one-shot per shell)
export PYTHONPATH=$PWD
```

A GPU is required for **trace generation** and **DeBERTa / probe training** but
not for stacking, baselines, or the statistical analyses.

For trace generation via the DeepSeek API instead of a local GPU:
```bash
cp .env.example .env
# Then put your DEEPSEEK_API_KEY in .env
```

---

## Quick start: 5-minute pilot

Verify the pipeline end-to-end on synthetic data (no GPU needed):

```bash
PYTHONPATH=. python scripts/validate_pipeline.py
```

This runs scoring -> segmentation -> parsing -> feature extraction -> JSONL
round-trip on synthetic traces and prints a green checklist if the codebase
is wired up correctly.

To reproduce the headline numbers, follow [`REPRODUCING.md`](REPRODUCING.md).

---

## Data and model availability

- **Raw traces** for the 8 dataset x model cells (~6,378 traces) are
  regenerable from `scripts/run_generation.sh`. We do not commit them since
  they are easily regenerated; the **parsed episodes**, all **feature CSVs**,
  all **OOF probability arrays**, and the **joined hybrid table** are committed
  under `data/` and `results/` for direct reproduction of the meta-learning
  results.
- **Step embeddings**, **graph artifacts**, and **hidden-state extractions**
  are large binaries (each up to ~165 MB) excluded from git via `.gitignore`.
  They are regeneratable with the corresponding `scripts/build_*.py` script.
  See `REPRODUCING.md` for details.

---

## Citing

If you build on this codebase, please cite the final report
(`reports/FINAL_REPORT.pdf`):

```
@misc{superhybrid2026,
  author = {Chen, Peng and Lin, Lixing and Ma, Youxuan},
  title  = {SuperHybrid: Single-Pass Black-Box Uncertainty Quantification for
            LLM Reasoning Traces},
  year   = {2026},
  note   = {Yale University, CPSC 4770/5770}
}
```

---

## License

Research code released for academic use. Model weights are under their
respective licenses (DeepSeek-R1: MIT; Llama 3.1: Meta Community License).
