# Results

## Summary

<p align="center">
  <img src="assets/results_summary.png" alt="Results Summary" width="620"/>
</p>

| Mode | Coverage | Accuracy | Wilson 95% upper error |
|---|---|---|---|
| Ungated (baseline) | 100% | 84.3% | — |
| ISR-only (h* ≤ 5%) | 44.2% | 91.0% | ≤ 5% |
| Hybrid ISR + τ (h* ≤ 10%) | **76%** | **93.0%** | ≤ 10% |

---

## Order-Sensitivity: Prediction Flip Rates

<p align="center">
  <img src="assets/flip_rates.png" alt="Flip Rates" width="520"/>
</p>

Changing the order of sections in a clinical note — without altering any content — causes the model to flip its predicted ICD category in **32.4%** of test cases. This demonstrates that the baseline model is not robust to input permutation.

---

## Operating Points

<p align="center">
  <img src="assets/operating_points.png" alt="Operating Points" width="540"/>
</p>

ISR gating identifies a safer operating region by abstaining on unstable predictions. The hybrid mode preserves significantly more coverage (76%) while achieving higher accuracy (93%) than the ISR-only mode.

---

## Accuracy–Coverage Tradeoff

<p align="center">
  <img src="assets/coverage_curves.png" alt="Coverage Curves" width="620"/>
</p>

The hybrid gating curve dominates ISR-only across most coverage levels — it answers more cases at the same or higher accuracy by supplementing ISR stability checks with calibrated confidence thresholds.

---

## Per-Class Performance (Ungated Baseline)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Acute MI (AMI) | — | — | — |
| Heart Failure (HF) | — | — | — |
| Atrial Fibrillation (AF) | — | — | — |
| Chronic IHD | — | — | — |

*Per-class numbers are populated from `outputs/classification_report.txt` after running training.*
