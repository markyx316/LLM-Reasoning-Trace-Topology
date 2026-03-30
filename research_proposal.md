# Research Proposal: Reasoning Trace Topology as Calibration-Free Uncertainty Quantification

## 1. Problem Statement and Motivation

Modern reasoning LLMs (DeepSeek-R1, OpenAI o-series, Qwen3, Claude with extended thinking) generate rich, structured chains of thought (CoT) before producing a final answer. These traces contain distinctive cognitive behaviors — backtracking, self-verification, sub-goal decomposition, rumination, and solution abandonment — that are visible in the surface-level text. A critical finding from the "Thoughtology" study (Marjanović et al., April 2025) is that on the AIME-24 benchmark, **correct solutions average ~2,000 tokens while incorrect ones average ~4,000 tokens**, and that rumination rates are negatively associated with accuracy. In other words, the *shape* of a reasoning trace carries a strong signal about whether the model is likely to be wrong.

Yet this signal is almost entirely untapped for the problem of **uncertainty quantification (UQ)** — arguably the most practically important open problem in LLM deployment. Current UQ methods fall into three categories, each with severe limitations:

1. **Logit-based methods** (perplexity, token entropy): Require white-box access to model internals. Increasingly unavailable as frontier models are API-only. Do not account for reasoning structure at all.
2. **Sampling-based methods** (semantic entropy, self-consistency, SelfCheckGPT): Require generating N≥5 independent responses per query. This multiplies inference cost by 5–20x — and for reasoning models that already generate thousands of tokens per response, this becomes prohibitively expensive.
3. **Verbalized confidence** (asking the model "how confident are you?"): Shown to be poorly calibrated. Frontier models exhibit systematic overconfidence. Anthropic and OpenAI's own research shows that RLHF incentivizes confident guessing over calibrated uncertainty.

**The core insight of this project:** A single reasoning trace, from a single generation, already encodes rich structural information about model uncertainty — if we know how to read it. Instead of asking the model for its confidence (unreliable), or sampling many responses (expensive), we can extract topological features from the trace's cognitive structure and use them to predict correctness.

---

## 2. Related Work and Positioning

### 2.1 Closest Related Work

**"Topology of Reasoning" (Minegishi et al., NeurIPS 2025):** Extracts reasoning *graphs* from **hidden-state representations** (not surface text) by clustering internal activations using K-means. Analyzes cyclicity, diameter, and small-world index. Key finding: reasoning models exhibit ~5 cycles per sample and 6x higher small-world index than base models. **Critical difference from our work:** This paper operates on internal model representations (requires white-box access), studies interpretability (not UQ), and does not predict per-instance correctness. Their graphs are representation-space objects; ours are text-surface cognitive-behavior objects.

**"The Shape of Reasoning" (2025, under review):** Applies topological data analysis (TDA) to reasoning traces. Uses persistent homology to compute Betti numbers and persistence diagrams. Key finding: topological features correlate more strongly with reasoning quality than standard graph metrics. **Critical difference:** Operates only on the AIME dataset, uses computationally expensive TDA machinery (limiting scalability), and frames the problem as quality assessment rather than actionable UQ (no comparison against UQ baselines, no selective generation, no calibration analysis).

**"Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology" (Da et al., KDD 2025):** Connects reasoning topology to uncertainty. **Critical difference:** Focuses on explanation uncertainty (whether the explanation is reliable), not answer correctness prediction. Uses a different graph construction method and does not compare against standard UQ baselines.

### 2.2 What This Project Adds

Our work is the first to:

1. Frame reasoning trace structure explicitly as a **UQ method** and benchmark it head-to-head against established UQ baselines (verbalized confidence, self-consistency, semantic entropy, perplexity).
2. Operate entirely at the **text surface level** — requiring only the generated text, not hidden states. This makes it applicable to black-box API models where traces are visible (e.g., DeepSeek-R1 exposes `<think>` blocks).
3. Evaluate using standard UQ metrics (AUROC for selective generation, Expected Calibration Error, Prediction Rejection Ratio).
4. Test **cross-domain generalization** of structural features (train on math, test on science reasoning).
5. Demonstrate practical utility via a **selective generation** pipeline: abstain on high-uncertainty traces, improving overall accuracy on answered questions.

---

## 3. Methodology

### 3.1 Models

**Primary model:** DeepSeek-R1-Distill-Qwen-7B (open-weight, MIT license, runs on a single GPU with 24GB VRAM, or free via HuggingFace Inference API). This model generates rich `<think>...</think>` blocks with visible reasoning.

**Secondary models (for generalization testing):**
- DeepSeek-R1-Distill-Qwen-14B (if compute allows)
- DeepSeek-R1-Distill-Llama-8B (different architecture family)
- Qwen3-4B with thinking mode enabled (different training recipe)

### 3.2 Datasets and Benchmarks

| Dataset | Domain | Size | Difficulty | Verifiability |
|---------|--------|------|------------|---------------|
| MATH500 | Math (diverse) | 500 | Mixed | Exact match |
| GSM8K | Grade-school math | 1,319 | Easy | Exact match |
| GPQA Diamond | Graduate science | 198 | Hard | Multiple choice |
| ARC-Challenge | Science reasoning | 1,172 | Medium | Multiple choice |

**Rationale:** MATH500 and GSM8K provide a difficulty gradient within math. GPQA and ARC test whether features learned on math transfer to science domains. AIME provides a high-difficulty stress test.

### 3.3 Trace Generation Protocol

For each dataset item:
1. Generate a single response with `temperature=0.6` (the recommended setting for DeepSeek-R1 reasoning tasks), `max_tokens=32768`.
2. Extract the full `<think>` block as the reasoning trace.
3. Extract the final answer from outside the `<think>` block.
4. Score correctness by exact-match against ground truth (for math) or answer-letter match (for MC).

**Additionally**, for the self-consistency baseline:
- Generate 8 independent responses per item at `temperature=0.8`.
- Compute majority-vote agreement and semantic clustering metrics.

### 3.4 Trace Parsing: From Text to Cognitive Structure

This is the methodological heart of the project. We parse each reasoning trace into a sequence of **cognitive episodes**, each tagged with a behavior type. The taxonomy is adapted from the DeepSeek-R1 Thoughtology (Marjanović et al., 2025) and the cognitive behavior framework of Gandhi et al. (2025):

**Behavior Taxonomy:**

| Behavior | Trigger Patterns (regex + heuristics) | Symbol |
|----------|--------------------------------------|--------|
| **Forward reasoning** | Default state; no trigger pattern (new claims, derivations, calculations) | F |
| **Verification** | "let me check", "let me verify", "to confirm", "double-check", "is this correct" | V |
| **Backtracking** | "wait", "actually", "no, that's wrong", "I made an error", "let me reconsider" | B |
| **Restart/Abandonment** | "let me try a different approach", "alternatively", "starting over", "another way" | R |
| **Sub-goal decomposition** | "first, I need to", "step 1:", "breaking this down", "the key insight is" | S |
| **Hesitation/Rumination** | "hmm", "I'm not sure", "this is tricky", repeated re-statement of the same fact | H |
| **Conclusion** | "therefore", "the answer is", "in conclusion", "so the final answer" | C |

**Parsing implementation:** A hybrid approach:
1. **Rule-based segmentation:** Split the trace at sentence boundaries (using spaCy or regex). Apply keyword/regex matching for each behavior type. Assign each sentence the highest-priority matching behavior.
2. **LLM-assisted refinement (optional enhancement):** For ambiguous segments, use a small model (e.g., Qwen2.5-0.5B) with a classification prompt to disambiguate. This is cheap (0.5B model, few tokens per classification) and can be run in batch.

**Output:** Each trace becomes a behavior sequence, e.g.: `[S, F, F, F, V, F, F, B, F, F, V, V, H, R, F, F, F, V, C]`

### 3.5 Feature Extraction

From each behavior sequence, extract the following feature groups:

**Group 1: Length and Proportion Features (6 features)**
- Total trace length (tokens)
- Total number of cognitive episodes
- Proportion of each behavior type (F, V, B, R, S, H) relative to total episodes

**Group 2: Structural/Topological Features (10 features)**
- **Backtrack count:** Total number of B episodes
- **Verification count:** Total number of V episodes
- **Restart count:** Total number of R episodes
- **Verification-to-Forward ratio:** V_count / F_count — measures how much the model checks vs. progresses
- **Backtrack position (normalized):** Mean position of B episodes as fraction of total length — do backtracks cluster early (exploratory) or late (corrective)?
- **First-correct-attempt position:** Position of the first C episode divided by total length — how early does the model first attempt an answer?
- **Verification clustering coefficient:** Are verifications evenly distributed or clustered? Compute the standard deviation of gaps between consecutive V episodes, normalized by trace length
- **Longest forward run:** Maximum number of consecutive F episodes without any V, B, or R — measures the longest stretch of uninterrupted forward progress
- **Behavior transition entropy:** Shannon entropy of the bigram transition matrix over the behavior sequence — high entropy = chaotic, unpredictable reasoning; low entropy = structured, regular reasoning
- **Cycle count (text-level):** Number of times the model returns to verify/reconsider a sub-problem it already addressed (detected by semantic similarity of problem references across episodes using sentence embeddings)

**Group 3: Content-Free Meta Features (4 features)**
- Ratio of "Wait"-family tokens to total tokens
- Number of question marks in the trace (self-questioning)
- Number of explicit negations ("no", "wrong", "incorrect", "that's not right")
- Token-level repetition rate (fraction of 4-grams that appear more than once)

**Total: ~20 features per trace.**

### 3.6 Predictive Modeling

**Task:** Binary classification — predict whether the final answer is correct (1) or incorrect (0).

**Models to try (in order of complexity):**
1. **Logistic Regression** (L2-regularized): Interpretable, fast, strong baseline. Feature coefficients directly tell us which structural properties predict failure.
2. **Random Forest** (100 trees): Captures non-linear interactions between features (e.g., high verification count is fine when backtracks are low, but bad when both are high).
3. **XGBoost**: For comparison; often marginal improvement over RF for tabular data.
4. **Single-feature baselines**: Test each feature individually to understand marginal predictive power.

**Training/evaluation protocol:**
- **Within-dataset:** 5-fold stratified cross-validation on each dataset separately. Report mean ± std of AUROC and AUPRC.
- **Cross-dataset transfer:** Train on MATH500, test on GPQA and ARC. This tests whether structural features of "uncertain reasoning" generalize across domains — the most scientifically interesting experiment.
- **Cross-model transfer:** Train on DeepSeek-R1-Distill-Qwen-7B traces, test on DeepSeek-R1-Distill-Llama-8B or Qwen3-4B traces. This tests whether reasoning topology as a UQ signal is model-agnostic.

### 3.7 Baselines for Comparison

| Baseline | Category | Requirements | Cost |
|----------|----------|-------------|------|
| Verbalized confidence | Self-report | Prompt the model to state confidence 0–100 after answering | 1 generation |
| Trace length (tokens) | Naive structural | Count tokens | 1 generation |
| Self-consistency (N=8) | Sampling | Generate 8 responses, majority vote agreement | 8 generations |
| Semantic entropy (N=8) | Sampling | Cluster 8 responses semantically, compute entropy | 8 generations |
| Token-level perplexity | Logit-based (white-box) | Mean log-probability of generated tokens | 1 generation + logits |
| **Our method** | **Structural (text-only)** | **Parse trace, extract features, classify** | **1 generation** |

**Key comparison axes:**
- **Accuracy (AUROC):** How well does each method rank correct vs. incorrect answers?
- **Cost:** Number of generations required (our method: 1; sampling methods: 8+)
- **Access requirements:** Our method: text-only (black-box compatible); perplexity: white-box; others: text-only
- **Calibration (ECE):** How well do predicted probabilities match empirical accuracy?

### 3.8 Application: Selective Generation

The practical endgame: use the UQ score to **abstain** on uncertain predictions. For each method, construct a selective generation curve:
- Sort predictions by confidence (descending).
- At each coverage level (fraction of questions answered), compute accuracy on the answered set.
- Plot accuracy vs. coverage.
- The best UQ method achieves the highest accuracy at every coverage level.

Report: **Accuracy@80% coverage** (answer the 80% most confident predictions, what accuracy do you get?) and **AUPRC** (area under the precision-recall curve, treating correctness as the positive class).

---

## 4. Expected Results and Hypotheses

**H1 (Main):** Structural features from a single reasoning trace will achieve AUROC ≥ 0.75 for correctness prediction on MATH500, competitive with self-consistency (N=8) which requires 8x the compute.

**H2 (Feature importance):** The verification-to-forward ratio and backtrack position will be the strongest individual predictors. High V/F ratio with late backtracks strongly predicts failure (the model "knew something was wrong" but kept going).

**H3 (Cross-domain transfer):** A classifier trained on MATH500 structural features will achieve AUROC ≥ 0.65 on GPQA Diamond without retraining, because the topology of uncertain reasoning (many backtracks, high verification clustering) is domain-general.

**H4 (Overthinking signal):** Traces that are both long AND have high verification clustering will be the strongest failure predictors — this is the "overthinking" signature, where the model verifies obsessively because it senses something is wrong but can't fix it.

**H5 (Baseline comparison):** Verbalized confidence will be the weakest baseline (known poor calibration). Trace length alone will be a surprisingly strong baseline (given the length–accuracy correlation found by Marjanović et al.), but our full feature set will significantly outperform it by capturing *why* a trace is long (productive exploration vs. unproductive rumination).

---

## 5. Detailed Timeline (12 Weeks)

### Phase 1: Infrastructure and Data Collection (Weeks 1–3)

**Week 1: Environment setup**
- Set up inference pipeline for DeepSeek-R1-Distill-Qwen-7B (using vLLM or HuggingFace Transformers with 4-bit quantization via bitsandbytes if GPU-limited).
- Download and prepare all 5 evaluation datasets. Implement exact-match scoring for each.
- Run a pilot of 50 items from MATH500 to verify the pipeline end-to-end.

**Week 2: Large-scale trace generation**
- Generate 1 primary trace per item across all datasets (~3,200 items total). At ~2,000 tokens per trace, this is ~6.4M tokens of generation — roughly 8–12 hours on a single A100 or 2–3 days on an A6000.
- Generate 8 self-consistency samples for MATH500 and GPQA (for baseline comparison). This is the expensive step: ~4,000 × 8 × 2,000 = ~64M tokens. Budget ~2–4 days.
- Store all traces in a structured JSON format with metadata (prompt, trace, answer, correctness label).

**Week 3: Trace parser development**
- Build the behavior parser. Start with regex-based rules for each of the 7 behavior types.
- Manually annotate ~100 traces (20 per dataset) to validate parser accuracy. Compute inter-annotator agreement (you annotate, then compare against parser output).
- Refine parser rules. Target ≥80% agreement with manual annotations on behavior type assignment.

### Phase 2: Feature Engineering and Modeling (Weeks 4–7)

**Week 4: Feature extraction**
- Implement all 20 features from Section 3.5.
- Run feature extraction on all generated traces.
- Perform exploratory data analysis: correlation matrix between features, distribution of each feature for correct vs. incorrect traces (box plots, histograms).

**Week 5: Baseline implementations**
- Implement all 6 baselines from Section 3.7.
- For verbalized confidence: re-prompt the model with each question + "After solving, rate your confidence from 0 to 100."
- For perplexity: extract token log-probabilities during the original generation (this is free with HuggingFace models).
- For semantic entropy: implement the clustering pipeline from Kuhn et al. (2023) using a DeBERTa-based NLI model.

**Week 6: Model training and within-dataset evaluation**
- Train logistic regression, random forest, and XGBoost classifiers on each dataset using 5-fold cross-validation.
- Compute AUROC, AUPRC, ECE for all methods (ours + all baselines) on all datasets.
- Perform ablation study: which feature group (length, structural, meta) contributes most?

**Week 7: Cross-domain and cross-model transfer**
- Train on MATH500 → test on GPQA, ARC-Challenge.
- Train on GSM8K (easy) → test on AIME (hard). Does structure-based UQ transfer across difficulty levels?
- (If time allows) Generate traces from DeepSeek-R1-Distill-Llama-8B on MATH500 and test cross-model transfer.

### Phase 3: Analysis and Writing (Weeks 8–12)

**Week 8: Deep analysis**
- Feature importance analysis: SHAP values for the random forest, logistic regression coefficients.
- Error analysis: examine the 20 traces where our method is most confidently wrong (predicts correct but wrong, or predicts wrong but correct). What structural patterns mislead the classifier?
- The "right for wrong reasons" analysis: among traces marked correct, identify those where the reasoning is incoherent but the answer is right. Does our topology method flag these?

**Week 9: Selective generation experiments**
- Generate accuracy vs. coverage curves for all methods.
- Compute Accuracy@80%, Accuracy@90% coverage.
- Estimate compute savings: "our method achieves X% of self-consistency's AUROC at 1/8 the inference cost."

**Week 10: Visualizations and figures**
- Create the main results table (AUROC across datasets × methods).
- Feature importance bar chart.
- Selective generation curves (the "money plot").
- Example traces: one correct trace with clean topology, one incorrect trace with chaotic topology, annotated with behavior labels.

**Weeks 11–12: Paper writing**
- Draft following standard ML paper structure: Introduction, Related Work, Method, Experiments, Analysis, Conclusion.
- Target length: 8–10 pages (NeurIPS/EMNLP format).
- Internal review and revision.

---

## 6. Compute Requirements

| Task | GPU Hours | Notes |
|------|-----------|-------|
| Primary trace generation (3,200 items) | 8–12 hrs | 1× A100 or 2× A6000 |
| Self-consistency traces (8 samples, 700 items) | 20–30 hrs | For baselines only |
| Classifier training | <1 hr | CPU only (tabular ML) |
| Feature extraction | <1 hr | CPU + small embedding model |
| **Total** | **~30–45 GPU-hours** | Feasible on university cluster or cloud ($50–100) |

**Alternatively:** Use the free DeepSeek API (rate-limited) or HuggingFace Inference API for generation if no local GPU is available. The self-consistency baseline is the most expensive part — consider reducing to N=4 if budget-constrained.

---

## 7. Key Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Behavior parser accuracy too low | Medium | Use LLM-assisted parsing (cheap with 0.5B model). Fall back to simpler heuristics (wait-token counting, sentence-level similarity). Even 70% parser accuracy may suffice since we aggregate into features. |
| Features don't outperform trace length alone | Medium | This is actually a publishable finding (the structural content of traces is redundant with length). Pivot to analyzing *why* length is such a strong signal and whether there exist subpopulations where structure adds value (e.g., hard problems where length is uninformative). |
| Cross-domain transfer fails | Medium-High | Report as a finding: structural features are domain-specific. Investigate which features transfer and which don't. This informs future work on universal UQ features. |
| Insufficient compute for all experiments | Low | Prioritize: (1) MATH500 main experiment, (2) GPQA transfer, (3) baselines. Drop ARC, AIME, and cross-model experiments if needed. The core story can be told with MATH500 + GPQA alone. |

---

## 8. Expected Contributions

1. **A new UQ paradigm for reasoning models:** Single-generation, text-surface, structure-based confidence estimation that is cheaper than sampling methods and more reliable than self-verbalized confidence.
2. **A reusable trace parsing toolkit:** Code for segmenting reasoning traces into cognitive episodes, applicable to any reasoning model with visible CoT.
3. **Empirical findings on the structure–accuracy relationship:** Which structural features most strongly predict failure? Do they generalize across domains and models?
4. **Practical selective generation results:** How much accuracy can you gain by abstaining on structurally-uncertain traces?

---

## 9. Publication Venues

- **Primary targets:** EMNLP 2026 (deadline ~June), NeurIPS 2026 (deadline ~May), ACL 2026 Rolling Review
- **Workshop alternatives:** ICML 2026 workshops on Reliable ML or Reasoning, NeurIPS 2025 workshops
- **The project scope is calibrated for a short (4-page) workshop paper at minimum, expandable to a full paper with all experiments.**

---

## 10. References (Key Papers)

1. Minegishi et al. (2025). "Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties." NeurIPS 2025.
2. Marjanović et al. (2025). "DeepSeek-R1 Thoughtology: Let's Think About LLM Reasoning." arXiv:2504.07128.
3. Gandhi et al. (2025). "Cognitive Behaviors in LLMs." [Defines behavior taxonomy for reasoning traces.]
4. Lin et al. (2023). "Generating with Confidence: Uncertainty Quantification for Black-box LLMs." TMLR 2024.
5. Kuhn et al. (2023). "Semantic Entropy." [Sampling-based UQ baseline.]
6. Vashurin et al. (2025). "Benchmarking UQ Methods for LLMs with LM-Polygraph." TACL 2025.
7. DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability via RL." arXiv:2501.12948.
8. Chen et al. (2024). "Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs."
9. Wang et al. (2025). "NoWait: Removing Thinking Tokens Improves Reasoning Efficiency."
10. Sui et al. (2025). "Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models."
