# How to Run

## Prerequisites

- Python 3.10+
- CUDA GPU (strongly recommended — Clinical-Longformer on CPU is very slow)
- Access to [MIMIC-IV-Ext Cardiac Disease dataset](https://physionet.org) (requires PhysioNet credentialing)

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure paths

Edit `config.py`:

```python
DATA_PATH  = "/path/to/MIMIC-CARDIAC-EXT/"   # folder with the 3 CSVs
OUTPUT_DIR = "./outputs/"                      # where model + results are saved
```

Required CSV files in `DATA_PATH`:

| File | Description |
|---|---|
| `heart_diagnoses.csv` | One row per admission with clinical text columns |
| `heart_diagnoses_all.csv` | All ICD codes per admission |
| `heart_labevents_first_lab.csv` | First-recorded lab values per admission |

## 3. Train the classifier

```bash
python train.py
```

This runs Phase 1: loads data, trains Clinical-Longformer for 8 epochs, and saves:
- `outputs/final_model/` — HuggingFace model + tokenizer
- `outputs/val_prob_matrix.csv` — validation predictions
- `outputs/test_prob_matrix.csv` — test predictions
- `outputs/classification_report.txt`

Training takes ~4–8 hours on a single A100.

## 4. Run ISR gating

```bash
python run_isr.py
```

This runs Phase 2 (no retraining needed):
- Calibrates temperature on the validation set
- Selects ISR threshold and hybrid (τ, γ) parameters
- Evaluates on the test set
- Saves `outputs/final_results.json` and `outputs/isr_coverage_curve.csv`

## 5. Regenerate charts (optional)

```bash
python docs/generate_charts.py
```

Outputs saved to `docs/assets/`.

---

## Override paths via CLI

```bash
python train.py --data_path /my/data --output_dir /my/outputs
python run_isr.py --model_path /my/outputs --h_star_isr 0.05 --h_star_hybrid 0.10
```
