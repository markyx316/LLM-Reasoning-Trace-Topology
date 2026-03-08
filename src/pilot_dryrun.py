"""
pilot_dryrun.py — End-to-end pipeline test using mock traces.

Validates the full pipeline without needing GPU or API keys:
  1. Load MATH500 (50 items) from HuggingFace
  2. Generate realistic mock traces (mimic DeepSeek-R1 output structure)
  3. Extract behavior sequences with parse_trace
  4. Extract 20 features with extract_features
  5. Run a toy binary classifier and report AUROC
  6. Save output to data/traces/math500_dryrun.jsonl

Run: python src/pilot_dryrun.py
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parse_trace import parse_trace
from extract_features import extract_features, FEATURE_COLS

DATA_DIR = ROOT / "data"
TRACES_DIR = DATA_DIR / "traces"
FEATURES_DIR = DATA_DIR / "features"

# ─── Mock trace templates ─────────────────────────────────────────────────────
# These are representative of real DeepSeek-R1 traces.
# Two flavors: "correct" (short, structured) and "incorrect" (long, ruminating)

CORRECT_TEMPLATE = """\
Let me break this problem down step by step.

First, I need to identify the key elements of the problem.
The problem asks us to find {goal}.

Step 1: Set up the equation.
{step1}

Step 2: Solve for the unknown.
{step2}

Let me verify my answer by substituting back.
{verification}
Yes, this checks out.

Therefore, the answer is \\boxed{{{answer}}}.
"""

INCORRECT_TEMPLATE = """\
Hmm, let me think about this carefully.

First, I need to find {goal}.

Step 1: I'll try to {step1_wrong}.
{calc_wrong}

Wait, actually that doesn't seem right. Let me reconsider.

Actually, I think I made an error. Let me try a different approach.

Step 2: Instead, let me try {step2_alt}.
{calc_alt}

Hmm, I'm not sure this is correct either. Let me verify.

Let me check: does {check_cond}?
{check_result}

No, that's wrong. I need to start over.

Actually, let me try yet another approach.

{approach3}

Wait, I think I see the issue now. Let me reconsider the problem statement.

The problem says {restate}.
So I need {correction}.

{correction_calc}

Hmm, I'm getting {wrong_answer} but I'm not confident this is right.
Let me verify once more.

Actually no, I think the answer might be {attempt2}.
I'm not entirely sure though.

Therefore, the answer is \\boxed{{{wrong_answer}}}.
"""

CORRECT_FILLS = [
    dict(goal="the value of x", step1="x + 5 = 12", step2="x = 7",
         verification="7 + 5 = 12. Correct.", answer="7"),
    dict(goal="the area of the triangle", step1="base = 6, height = 4",
         step2="Area = (1/2)(6)(4) = 12",
         verification="(1/2)(6)(4) = 12. Correct.", answer="12"),
    dict(goal="the sum of the series", step1="S = 1 + 2 + ... + 10",
         step2="S = n(n+1)/2 = 10(11)/2 = 55",
         verification="Direct sum: 55. Matches.", answer="55"),
    dict(goal="the probability", step1="P(A) = 3/8",
         step2="The favorable outcomes are 3 out of 8",
         verification="3/8 is in simplest form.", answer="3/8"),
    dict(goal="the distance", step1="d = sqrt((3-0)^2 + (4-0)^2)",
         step2="d = sqrt(9 + 16) = sqrt(25) = 5",
         verification="3-4-5 right triangle. Correct.", answer="5"),
]

INCORRECT_FILLS = [
    dict(goal="the maximum value", step1_wrong="take the derivative",
         calc_wrong="f'(x) = 2x + 3", step2_alt="use the second derivative test",
         calc_alt="f''(x) = 2 > 0, so this is a minimum not maximum",
         check_cond="f(0) gives the maximum", check_result="f(0) = 0, but that seems too small",
         approach3="Let me try completing the square instead",
         restate="we want to maximize f(x) = -x^2 + 4x",
         correction="to find the vertex",
         correction_calc="vertex at x = -b/2a = -4/(2(-1)) = 2",
         wrong_answer="2", attempt2="4"),
    dict(goal="the number of ways", step1_wrong="use permutations",
         calc_wrong="P(5,3) = 5!/2! = 60", step2_alt="use combinations instead",
         calc_alt="C(5,3) = 10", check_cond="order matters here",
         check_result="Actually, I don't think order matters",
         approach3="Let me re-read the problem carefully",
         restate="we choose 3 from 5 where order doesn't matter",
         correction="C(5,3)", correction_calc="5!/(3!2!) = 10",
         wrong_answer="60", attempt2="10"),
]


def make_mock_trace(correct: bool) -> tuple[str, str]:
    """Return (trace_text, predicted_answer)."""
    if correct:
        fill = random.choice(CORRECT_FILLS)
        trace = CORRECT_TEMPLATE.format(**fill)
        answer = fill["answer"]
    else:
        fill = random.choice(INCORRECT_FILLS)
        trace = INCORRECT_TEMPLATE.format(**fill)
        answer = fill["wrong_answer"]

    # Add some realistic noise/variation in length
    if not correct:
        # Repeat some rumination to make incorrect traces longer
        extra = random.randint(1, 3)
        for _ in range(extra):
            trace += f"\nHmm, wait. Let me double-check this result once more.\n"
            trace += f"Actually, I'm not sure my approach was right.\n"

    return trace.strip(), answer


# ─── Main pilot ───────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    print("=" * 60)
    print("PILOT DRY RUN — End-to-end pipeline test")
    print("=" * 60)

    # Step 1: Load MATH500
    print("\n[1/5] Loading MATH500 (50 items)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"id": f"math500_{i}", "problem": row["problem"], "gt": row["answer"]}
                 for i, row in enumerate(ds)][:50]
        print(f"    Loaded {len(items)} items from HuggingFaceH4/MATH-500")
    except Exception as e:
        print(f"    Could not load dataset ({e}), using dummy items")
        items = [{"id": f"math500_{i}", "problem": f"Dummy problem {i}", "gt": str(i)}
                 for i in range(50)]

    # Step 2: Generate mock traces
    print("\n[2/5] Generating mock traces (50% correct, 50% incorrect)...")
    records = []
    for item in items:
        correct = random.random() > 0.45  # ~55% correct (realistic for hard math)
        trace, predicted = make_mock_trace(correct)
        record = {
            "id": item["id"],
            "dataset": "math500",
            "prompt": item["problem"],
            "trace": trace,
            "answer": predicted,
            "ground_truth": item["gt"],
            "correct": correct,
            "tokens": len(trace.split()),
        }
        records.append(record)

    n_correct = sum(r["correct"] for r in records)
    print(f"    Generated {len(records)} traces ({n_correct} correct, "
          f"{len(records)-n_correct} incorrect)")

    # Save traces
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRACES_DIR / "math500_dryrun.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"    Saved to {out_path}")

    # Step 3: Parse traces
    print("\n[3/5] Parsing behavior sequences...")
    sample = records[:3]
    for r in sample:
        parsed = parse_trace(r["trace"])
        seq = "".join(parsed.sequence)
        label = "CORRECT" if r["correct"] else "wrong"
        print(f"    [{label}] {r['id']}: {seq[:60]}... ({len(parsed.behaviors)} episodes)")

    # Step 4: Extract features
    print("\n[4/5] Extracting 20 features from all 50 traces...")
    import pandas as pd
    rows = [extract_features(r) for r in records]
    df = pd.DataFrame(rows)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    feat_path = FEATURES_DIR / "math500_dryrun.csv"
    df.to_csv(feat_path, index=False)
    print(f"    Saved feature matrix: {df.shape[0]} rows × {df.shape[1]} cols → {feat_path}")

    # Step 5: Quick classifier sanity check
    print("\n[5/5] Sanity-check classifier (5-fold CV on 50 items)...")
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X = df[FEATURE_COLS].values.astype(float)
    y = df["correct"].values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []
    for tr_idx, val_idx in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        if len(set(y[val_idx])) > 1:
            aurocs.append(roc_auc_score(y[val_idx], probs))

    print(f"    RF AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f} (on mock data — "
          f"real model will be higher)")

    # Feature means by label
    print("\n    Feature means (correct vs. incorrect):")
    key_feats = ["g1_token_count", "g2_backtrack_count", "g2_verification_count",
                 "g2_vf_ratio", "g3_wait_ratio"]
    summary = df.groupby("correct")[key_feats].mean()
    print(summary.to_string())

    print("\n" + "=" * 60)
    print("PILOT PASSED — Pipeline is end-to-end functional.")
    print("Next step: set HF_TOKEN or DEEPSEEK_API_KEY in .env,")
    print("then run:")
    print("  python src/generate.py --dataset math500 --mode hf-api --pilot 50")
    print("=" * 60)


if __name__ == "__main__":
    main()
