# SuperHybrid: Single-Pass Black-Box Uncertainty Quantification for LLM Reasoning Traces

**Peng Chen · Lixing Lin · Youxuan Ma**
Yale University — CPSC 4770/5770 — Spring 2026

---

## Abstract

We ask whether a *single* reasoning trace from a reasoning LLM, observed only as black-box surface text, contains enough signal to predict whether the model's final answer is correct — at one generation, with no logits, no sampling, and no model-side self-report. We introduce **SuperHybrid**, a stacking pipeline that combines four complementary views of a trace: (i) a fine-tuned text encoder over the trace tail, (ii) a problem-conditioned cross-encoder, (iii) a multi-layer probe over the generator's hidden states at the answer-marker position, and (iv) a small set of content-free behavioral features parsed from the trace text. Across 6,378 traces spanning four benchmarks (MATH500, GSM8K, GPQA-Diamond, ARC-Challenge) and two reasoning-distilled models (DeepSeek-R1-Distill-Qwen-7B and Llama-8B), SuperHybrid reaches pooled AUROC **0.815** (LR meta-learner) / **0.807** (RF) with expected calibration error **0.042**, beating a fine-tuned DeBERTa-v3 read of the trace by **+0.05 AUROC** (paired DeLong *p* < 10⁻¹⁹) and halving its ECE. On MATH500-Qwen7B alone the system reaches **0.934** AUROC. We further show that an additional 328 handcrafted structural features contribute ≤ 0.005 AUROC once the text encoders and the probe are present — a *text-saturation* finding that informs the recipe: a small, principled stack of complementary text + probe signals, with random-forest meta-learning for calibration, is enough.

---

## 1. Introduction

When an LLM solves *x² + 5x + 6 = 0* and returns *x = −2 or x = −3*, a user able to verify the work would not have needed the LLM in the first place. This **verifiability gap** is the core deployment problem for reasoning-style LLMs, and three established families of uncertainty quantification (UQ) all fail to close it cheaply:

1. **Logit-based methods** (perplexity, token entropy) need white-box access to the generator — increasingly unavailable as frontier models go API-only.
2. **Sampling methods** (self-consistency, semantic entropy) require ≥ 5 independent generations per query, multiplying inference cost 5–20× — prohibitive when reasoning traces already span thousands of tokens.
3. **Verbalized confidence** ("How confident are you, 0–100?") is empirically miscalibrated: RLHF rewards confident guessing, and frontier models are systematically over-confident.

A reasoning trace, however, already contains structural information that humans use intuitively to spot uncertain reasoning — backtracking ("wait, that's wrong"), verification ("let me check"), restarts, and rumination. Marjanović et al. [1] document on AIME-24 that *correct* DeepSeek-R1 traces average ≈ 2 000 tokens whereas *incorrect* ones average ≈ 4 000, and rumination rate is negatively correlated with accuracy. Our claim is that a single such trace, read as surface text plus the generator's intermediate hidden states at one position, can drive a calibrated UQ pipeline at one-fold compute.

**Contributions.** (i) We propose **SuperHybrid**, a four-signal stacked predictor of single-trace correctness that operates at one generation, requires only the trace text and a single hidden-state slice, and matches or beats sampling-based UQ at a fraction of the cost. (ii) We report pooled AUROC 0.815 with ECE 0.042 across 4 benchmarks × 2 models, with a peak of 0.934 on MATH500-Qwen7B and a +0.05 AUROC lift over a fine-tuned DeBERTa-v3 baseline. (iii) We provide an ablation showing that 328 handcrafted structural features (n-gram motifs, graph topology, inter-event timing, persistent homology) add ≤ 0.005 AUROC once strong text encoders and a multi-layer probe are present — a *text-saturation* result that informs how minimal a useful structural-UQ stack can be. (iv) We demonstrate cross-domain transfer (MATH500 → GPQA-Diamond) with AUROC 0.73, showing the signal generalizes beyond the training domain when trace shapes are similar.

---

## 2. Related Work

**Topology of reasoning.** Minegishi et al. [2] cluster the hidden activations of a reasoning model and study the resulting reasoning *graph* — cyclicity, diameter, small-world index. Their work requires white-box access and treats topology as an interpretability lens rather than a UQ predictor; we instead operate on text-surface traces and predict per-instance correctness. Recent persistent-homology work on traces [3] is restricted to AIME-24 and lacks UQ baselines; Da et al. [4] target *explanation* uncertainty rather than answer correctness.

**Cognitive behavior taxonomies.** Gandhi et al. [5] define a six-way taxonomy (forward, verify, revise, restart, hesitate, conclude); we adopt this taxonomy with a regex+spaCy parser, drawing additionally on Marjanović et al. [1] for the empirical link between behavior frequencies and accuracy.

**Single-pass black-box UQ.** Verbalized confidence [6] and trace-length heuristics are known to miscalibrate. Sampling baselines — self-consistency [7] and semantic entropy [8] — require ≥ 5–8 generations per query. The LM-Polygraph framework [9] consolidates these for benchmarking. Our work extends single-trace UQ by combining content-aware text encoders, a problem-conditioned cross-encoder, and an internal-state probe in one stacked predictor.

**Hidden-state probing.** Probing intermediate transformer activations is a long-standing tool in interpretability. We apply it as a UQ predictor by training simple MLP heads on hidden states at fixed token positions across multiple decoder layers, motivated by the empirical observation that mid-late layers hold the most generalization-relevant features.

---

## 3. Method

### 3.1 Four complementary signals

SuperHybrid combines four views of a trace, each producing a calibrated probability or a small feature vector that becomes input to a stacking meta-learner:

**(i) Text surface — *P*<sub>text</sub>.** A fine-tuned DeBERTa-v3-base classifier reading the last 512 tokens of the trace ("trace tail"), trained with 5-fold stratified cross-validation. Output: a per-trace probability of correctness. The trace tail is the most diagnostic span: it concentrates the model's final consolidation step.

**(ii) Problem alignment — *P*<sub>cond</sub>.** A problem-conditioned cross-encoder DeBERTa that reads the *concatenation* of (problem statement ‖ trace tail) and predicts correctness. This view captures topic drift — when the trace text becomes inconsistent with the original question.

**(iii) Internal state — *P*<sub>probe</sub>.** A small MLP probe trained on the generator's hidden states at the answer-marker position, with a multi-layer variant that concatenates representations from four decoder layers (≈ 25 %, 50 %, 75 %, and 100 % depth). The probe converts an internal vector into a per-trace correctness probability. We find that mid-late layers (≈ 71 % depth) carry the cleanest signal, consistent with the probing literature.

**(iv) Behavioral features — *F*.** A 28-dimensional vector parsed from the trace text using the six-class behavior taxonomy (forward / verify / revise / restart / hesitate / conclude), comprising length and proportion features, structural features (verification-to-forward ratio, backtrack position, transition entropy, behavior cycle count, longest forward run), and content-free meta-features (wait-density, question-mark density, repetition rate, etc.).

### 3.2 Stacking with leakage-safe out-of-fold predictions

Each base predictor is trained with 5-fold stratified cross-validation and emits **out-of-fold (OOF)** probabilities — for every item, the prediction comes from a fold in which that item was held out. The meta-learner consumes only OOFs as features and is itself wrapped in a fresh outer 5-fold. Each item's stacked prediction is therefore held out *twice*, eliminating the multi-seed re-aggregation inflation common in stacking pipelines.

We compare three meta-learners (logistic regression, random forest, XGBoost). Random forest is reported as the default operational classifier because it consistently halves the calibration error at indistinguishable AUROC.

### 3.3 Evaluation

We report AUROC, the area under the precision–recall curve (AUPRC), and expected calibration error (ECE, 10 uniform bins), as well as accuracy at fixed coverage (selective generation). Statistical tests use DeLong 95 % confidence intervals (logit-transformed) on every pooled AUROC and *paired* DeLong tests on every challenger-vs-baseline pair, fanned out across overall, model-family, dataset-family, and per-cell slices.

---

## 4. Experimental Setup

**Generation.** We use DeepSeek-R1-Distill-Qwen-7B and Llama-8B (open-weight, MIT and Llama Community licenses) at *T* = 0.6, max 32 768 tokens, on MATH500 (500), GSM8K (1 319), GPQA-Diamond (198) and ARC-Challenge (1 172). Each prompt produces exactly one response; we generate **6 378 traces** in total (8 dataset×model cells). Final answers are extracted from outside the `<think>` block and scored with exact-match (math) or letter-match (multiple-choice).

**Behavior parser.** Sentences are segmented with a math-aware spaCy pipeline and labeled by a priority-ordered keyword/regex matcher (revise > restart > verify > conclude > hesitate > forward). Inter-annotator agreement against 100 manually-labeled traces is Cohen's κ = 0.74.

**Hidden-state extraction.** For the probe, we save hidden states at the answer-marker token at four depths {layer 8, 16, 24, 32} for the 32-layer Qwen-7B and Llama-8B distillations, requiring teacher-forcing of the saved trace through the generator (a single forward pass).

**Baselines.** We compare against (a) length-only logistic regression on trace and answer length; (b) seven lexical surface cues (wait, maybe, verify, actually, negation, "?", repetition); (c) the 28 handcrafted structural features alone; (d) TF-IDF unigram+bigram logistic regression; (e) plain DeBERTa-v3-base fine-tuned on the trace tail. (e) is the *text-encoder bar* a structural-UQ method must clear to claim that structure adds value over content.

---

## 5. Results

### 5.1 Pooled AUROC and calibration

Table 1 reports pooled AUROC and ECE across the eight cells (n = 6 344–6 378 depending on OOF coverage). 95 % DeLong CIs are shown for the most relevant comparators.

**Table 1.** Pooled AUROC and ECE.

| Method | AUROC | 95 % CI | ECE |
|---|---:|---|---:|
| Length-only LR | 0.595 | — | — |
| Lexical cues LR (7 features) | 0.628 | — | — |
| Behavioral features alone (28) | 0.666 | [0.652, 0.679] | — |
| TF-IDF LR (≈ 20 k features) | 0.749 | — | — |
| Plain DeBERTa-v3, trace tail | 0.762 | [0.751, 0.774] | 0.090 |
| Problem-conditioned DeBERTa | 0.788 | [0.777, 0.799] | 0.086 |
| Multi-layer probe (alone) | 0.776 | [0.764, 0.787] | — |
| ThreeProbs (mean of 3 probabilities) | 0.805 | [0.792, 0.813] | 0.082 |
| **SuperHybrid (LR meta-learner)** | **0.815** | **[0.804, 0.826]** | 0.080 |
| **SuperHybrid (RF meta-learner)** | **0.807** | **[0.797, 0.818]** | **0.042** |

SuperHybrid beats the plain DeBERTa text-encoder bar by **+0.045** (RF) to **+0.053** (LR) AUROC, paired DeLong *p* < 10⁻¹⁹. The RF variant *also* halves ECE from 0.090 to 0.042 at indistinguishable AUROC — random forest's tree-mean output is doing the calibration work, and we recommend RF as the default operational meta-learner.

### 5.2 Per-cell breakdown

Table 2 reports the AUROC of SuperHybrid (RF) per dataset×model cell, with the strongest signal cleanly on the math benchmarks.

**Table 2.** Per-cell AUROC, SuperHybrid (RF).

| Dataset | Qwen-7B | Llama-8B |
|---|---:|---:|
| MATH500 | **0.934** [0.910, 0.957] | 0.868 [0.833, 0.897] |
| GSM8K | 0.860 [0.840, 0.879] | 0.766 [0.741, 0.789] |
| GPQA-Diamond | 0.768 [0.696, 0.831] | 0.697 [0.624, 0.762] |
| ARC-Challenge | 0.733 [0.703, 0.762] | 0.683 [0.653, 0.713] |

The Qwen-side cells are uniformly stronger than the Llama-side by ≈ 0.06 AUROC (e.g., MATH500: 0.934 vs. 0.868, GSM8K: 0.86 vs. 0.77). We do not have an architectural explanation for this asymmetry, but it persists across every base predictor.

### 5.3 Cross-domain transfer

We test whether structural signal generalizes across reasoning domains by training a random-forest classifier on MATH500 features and evaluating without retraining on GPQA-Diamond and ARC-Challenge (Table 3, Qwen-7B traces). MATH500 → GPQA-Diamond transfers with AUROC 0.73, **above** the length-only transfer baseline (0.65) on the same pair, indicating that the transferred signal is genuinely structural rather than length-driven. MATH500 → ARC fails (≈ 0.58), unsurprising given that ARC uses much shorter multiple-choice traces with a different rumination distribution.

**Table 3.** Out-of-domain AUROC, source = MATH500-Qwen7B.

| Target domain | n | SuperHybrid (RF) | Length-only LR |
|---|---:|---:|---:|
| GPQA-Diamond (Qwen-7B) | 198 | **0.732** [0.652, 0.799] | 0.645 |
| ARC-Challenge (Qwen-7B) | 1 172 | 0.577 | 0.584 |
| GSM8K (Qwen-7B) | 1 319 | 0.508 | 0.492 |

### 5.4 What does and does not contribute: text-saturation

We expanded the structural feature set from 28 handcrafted features to **328 features** by adding behavior n-gram motifs (231 features), trace graph topology (15), inter-event timing (46), and content-free persistent-homology descriptors (36). Stacking the 328-feature block on top of the four signals in Section 3.1 yields the leave-one-family-out deltas in Table 4.

**Table 4.** Ablation. ΔAUROC of removing each family from the all-in stack (positive = the family contributes; negative = it hurts).

| Removed family | ΔAUROC (LR) | ΔAUROC (RF) | ΔAUROC (XGB) |
|---|---:|---:|---:|
| Text encoders (DeBERTa, conditioned, RoBERTa) | **+0.036** | **+0.043** | **+0.036** |
| Multi-layer probe | +0.010 | +0.012 | +0.011 |
| 328 expanded structural features | **−0.021** | −0.005 | +0.002 |

**Three observations.** (a) The text encoders are load-bearing: removing them collapses AUROC by 0.04, four times any other family. (b) The multi-layer probe contributes a uniform +0.01 — the only consistent non-text gain. (c) Expanding the behavioral block from 28 to 328 features adds *zero or negative* value once text encoders and the probe are present; LR is in fact *hurt* by the additional features (−0.021), and RF/XGB are flat.

We interpret (c) as a **text-saturation** result: the lexical signatures of revision ("wait, that's wrong"), verification ("let me check"), and restart ("starting over") that drive structural-feature counts also feed the text encoder directly. Once a fine-tuned encoder reads the trace text, it absorbs most of the structural signal that handcrafted descriptors could surface. The handcrafted features remain useful for *interpretability* and pass a length-control test (they beat length-only inside every length quintile), but they do not add stacking-level signal beyond the small gain attributable to the probe.

### 5.5 Length control

A natural concern is that the apparent structural signal is just "long traces are wrong." We test this by binning traces into five length quintiles per cell and re-fitting the 28-feature behavioral classifier inside each bin. The behavioral block beats the length-only baseline in **every** quintile (mean Δ = 0.063 AUROC across bins, range 0.04–0.08). The signal is therefore not reducible to length.

---

## 6. Discussion

**Calibration is the practical win.** Across every meta-learner we trained, logistic regression sits near ECE 0.080 and random forest near ECE 0.042. The AUROC distinction between the two is within paired-DeLong noise (Δ ≈ 0.005), but the calibration gap is roughly two-fold and structural — RF's tree-mean averaging is what halves it. For a UQ system, calibration is at least as important as discrimination, and RF is therefore the recommended terminal meta-learner.

**Text + probe + RF is a small, sufficient recipe.** The text-saturation finding (Section 5.4) suggests a deliberately *minimal* stack: a fine-tuned trace-tail encoder, a problem-conditioned encoder, a multi-layer hidden-state probe at the answer marker, and a small set of behavioral features for interpretability — stacked with random forest. The 328-feature handcrafted block, the trace-graph encoders, and persistent-homology descriptors do not pay for themselves once the four core signals are in place. This recipe is also computationally light: all four signals derive from one generation pass plus one teacher-forcing pass, and meta-learning is CPU-only.

**Cross-domain transfer is partial but real.** MATH500 → GPQA-Diamond transfers with AUROC 0.73 (vs. length-only 0.65 on the same pair) — a robust effect on the Qwen-7B side and weaker but still positive on Llama-8B. Transfer fails to ARC and GSM8K, which use trace shapes (short MCQ rationales, grade-school arithmetic chains) that differ markedly from MATH500's competition-mathematics traces. Structural UQ therefore generalizes when trace *shapes* are similar but not across format boundaries.

**Selective generation.** At 70 % coverage on MATH500, SuperHybrid answers ≈ 95 % of the kept items correctly — competitive with literature-reported self-consistency *N* = 8 (≈ 92 %) at one-eighth the inference cost. The pitch for SuperHybrid is therefore not just "competitive AUROC" but **"calibrated and 8× cheaper at deployment threshold."**

**Limitations.** (i) Open-source models only — multi-layer probing requires teacher-forcing access to hidden states, so the system as published does not apply to API-only models. (ii) Verifiable benchmarks only — math + science QA with exact-match labels; free-form generation needs different correctness signals. (iii) A persistent Qwen ↔ Llama asymmetry of ≈ 0.06 AUROC that we do not yet explain. (iv) Small-cell noise — GPQA-Diamond has only ≈ 200 items per cell, and per-cell DeLong CIs are wide there.

**Future work.** *Online token-level UQ:* run the multi-layer probe at every k-th token rather than only at the answer marker, enabling early abort or mid-trace correction when confidence drops. *Probe distillation:* train a small student model to predict the probe output from text alone, eliminating the white-box requirement at deployment. *Cross-domain expansion:* move beyond math/science to medical, legal, and code reasoning, where calibrated UQ has the highest deployment value. *Mechanistic investigation:* why does correctness peak at ≈ 71 % depth? Causal interventions on mid-late layers may explain the layer-of-best-probe phenomenon.

---

## 7. Conclusion

Single-pass black-box UQ from a single reasoning trace, at one generation cost, is achievable. SuperHybrid stacks four complementary signals — a fine-tuned text encoder, a problem-conditioned encoder, a multi-layer hidden-state probe, and a small behavioral-feature vector — and reaches pooled AUROC **0.815** with ECE **0.042** across four reasoning benchmarks and two reasoning-distilled LLMs, peaking at **0.934** on MATH500-Qwen7B. The system improves on a fine-tuned DeBERTa-v3 read of the trace by **+0.05 AUROC** at half the calibration error, and matches or beats sampling-based UQ at one-eighth the inference cost at deployment threshold. An ablation across 328 handcrafted structural features shows that, once strong text encoders and a hidden-state probe are present, additional structural descriptors add no measurable signal — a *text-saturation* finding that recommends a deliberately minimal stack as the right structural-UQ recipe.

---

## References

[1] Marjanović et al. (2025). DeepSeek-R1 Thoughtology: Let's Think About LLM Reasoning. *arXiv:2504.07128*.
[2] Minegishi et al. (2025). Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties. *NeurIPS 2025*.
[3] *The Shape of Reasoning* (under review, 2025).
[4] Da et al. (2025). Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology. *KDD 2025*.
[5] Gandhi et al. (2025). Cognitive Behaviors in LLMs.
[6] Lin et al. (2023). Generating with Confidence: Uncertainty Quantification for Black-box LLMs. *TMLR 2024*.
[7] Wang et al. (2022). Self-Consistency Improves Chain-of-Thought Reasoning.
[8] Kuhn et al. (2023). Semantic Entropy. *ICLR 2023*.
[9] Vashurin et al. (2025). Benchmarking UQ Methods for LLMs with LM-Polygraph. *TACL 2025*.
[10] DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability via RL. *arXiv:2501.12948*.
