# Architecture

## System Overview

The pipeline has two sequential phases:

```
MIMIC-IV-Ext CSVs
       │
       ▼
┌─────────────────────────────┐
│  Phase 1 — Classifier       │
│  MIMICCardiacDataProcessor  │
│  → Clinical text builder    │
│  → Subject-level splits     │
│  → Clinical-Longformer      │
│     fine-tune (8 epochs)    │
└────────────┬────────────────┘
             │  val_prob_matrix.csv
             │  test_prob_matrix.csv
             ▼
┌─────────────────────────────┐
│  Phase 2 — ISR Gating       │
│  Temperature calibration    │
│  → Validation ISR table     │
│  → Threshold selection      │
│  → Test evaluation          │
└─────────────────────────────┘
```

---

## Phase 1: Clinical-Longformer Classifier

**Model:** `yikuan8/Clinical-Longformer` (4096-token context)

**Input format:**

```
[EVIDENCE] Laboratory Results: troponin: 2.3 ng/mL; ECG: ST elevation V1-V4
[CONTEXT] Chief Complaint: chest pain  History: 2h chest pain radiating to arm
```

Sections are tagged with `[EVIDENCE]` (objective findings) and `[CONTEXT]` (subjective history). Special tokens are added to the tokenizer vocabulary and receive global attention.

**Training details:**

| Parameter | Value |
|---|---|
| Model | Clinical-Longformer |
| Max sequence length | 4096 |
| Optimizer | AdamW (lr=2e-5, wd=0.01) |
| Batch size | 1 (grad accum = 8) |
| Epochs | 8 |
| Loss | Weighted cross-entropy |
| Split | 70/15/15 (subject-level) |

**Label mapping (ICD precedence order):**

| Label | ICD Prefixes |
|---|---|
| `acute_mi` | I21, I22 |
| `heart_failure` | I50 |
| `atrial_fib` | I48 |
| `chronic_ihd` | I25 |

---

## Phase 2: ISR Gating

### Information Sufficiency Ratio

For a clinical note **x** with predicted class **ŷ**:

1. Compute `p_full = P(ŷ | x)` using raw softmax
2. Generate **m=6** section-level permutations `{x_k}`
3. For each permutation: `δ_k = clip(log p_full − log P(ŷ | x_k), 0, B)`
4. Estimate robust prior: `q̂ = quantile(P(ŷ | x_k), 0.25)`
5. Compute KL divergence term: `b2t = p* log(p*/q̂) + (1−p*) log((1−p*)/(1−q̂))`
6. **ISR = δ̄ / b2t** where δ̄ is the trimmed mean of deltas

**Accept if ISR ≥ threshold** (threshold selected on validation to meet Wilson error bound ≤ h*).

### Hybrid Gating

Extends ISR-only by also accepting cases where the *calibrated* model is highly confident:

```
Accept if:  (ISR ≥ threshold)
         OR (p_cal ≥ τ  AND  margin_cal ≥ γ)
```

τ and γ are jointly grid-searched on validation to maximise coverage subject to the same Wilson error guarantee.

### Temperature Calibration

Platt scaling via LBFGS minimises NLL on validation logits. The calibrated temperature T is used only for the hybrid confidence gate — ISR itself always uses raw (T=1) probabilities.
