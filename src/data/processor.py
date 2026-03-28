"""
Data loading and preprocessing for MIMIC-IV Cardiac Disease dataset.

Expected CSVs in DATA_PATH:
  heart_diagnoses.csv           – one row per admission with clinical text columns
  heart_diagnoses_all.csv       – all ICD codes per admission
  heart_labevents_first_lab.csv – first-recorded lab values per admission
"""

import os
import re
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from config import CONTEXT_HEADERS, EVIDENCE_HEADERS, SEED


class MIMICCardiacDataProcessor:
    _LAB_KEYS = [
        ("troponin", r"troponin|trop\b|tni|tnt"),
        ("ckmb", r"ck-?mb|cpk"),
        ("bnp", r"\bnt-?probnp\b|\bbnp\b"),
        ("creatinine", r"\bcreatinine\b"),
        ("sodium", r"\bsodium\b|\bna\b"),
        ("potassium", r"\bpotassium\b|\bk\b"),
        ("hemoglobin", r"\bhemoglobin\b|\bhgb?\b"),
    ]

    _BUCKETS = {
        "acute_mi": ("I21", "I22"),
        "heart_failure": ("I50",),
        "atrial_fib": ("I48",),
        "chronic_ihd": ("I25",),
    }
    _PRECEDENCE = ["acute_mi", "heart_failure", "atrial_fib", "chronic_ihd"]

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_data(self) -> None:
        print("Loading CSVs...")
        self.diagnoses_df = pd.read_csv(os.path.join(self.data_path, "heart_diagnoses.csv"))
        self.icd_df = pd.read_csv(os.path.join(self.data_path, "heart_diagnoses_all.csv"))
        self.labs_df = pd.read_csv(os.path.join(self.data_path, "heart_labevents_first_lab.csv"))

        for df in (self.diagnoses_df, self.icd_df, self.labs_df):
            df.columns = df.columns.str.strip()

        self.icd_df["icd_code"] = self.icd_df["icd_code"].astype(str)
        self.diagnoses_df["subject_id"] = self.diagnoses_df["subject_id"].astype(int)
        self.diagnoses_df["hadm_id"] = self.diagnoses_df["hadm_id"].astype(int)
        self.icd_df["hadm_id"] = self.icd_df["hadm_id"].astype(int)

        print(
            f"  diagnoses={self.diagnoses_df.shape}  "
            f"icd_all={self.icd_df.shape}  "
            f"labs={self.labs_df.shape}"
        )

    # ── Labels ────────────────────────────────────────────────────────────────

    def create_labels(self) -> pd.DataFrame:
        print("Mapping ICD codes to cardiac labels...")
        hadm_to_bucket: dict[int, str] = {}
        for hadm_id, grp in self.icd_df.groupby("hadm_id"):
            hits = set()
            for code in grp["icd_code"].astype(str):
                for bucket, prefixes in self._BUCKETS.items():
                    if any(code.startswith(p) for p in prefixes):
                        hits.add(bucket)
            for bucket in self._PRECEDENCE:
                if bucket in hits:
                    hadm_to_bucket[hadm_id] = bucket
                    break

        self.diagnoses_df["cardiac_label"] = (
            self.diagnoses_df["hadm_id"].map(hadm_to_bucket).fillna("unlabeled")
        )
        self.diagnoses_df = self.diagnoses_df[
            self.diagnoses_df["cardiac_label"] != "unlabeled"
        ].copy()
        self.diagnoses_df["label"] = self.label_encoder.fit_transform(
            self.diagnoses_df["cardiac_label"]
        )

        print(self.diagnoses_df["cardiac_label"].value_counts().to_string())
        print(f"  Total labeled: {len(self.diagnoses_df)}")
        return self.diagnoses_df

    # ── Clinical text construction ────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        if pd.isna(text):
            return ""
        t = str(text).replace("___", "[DEIDENTIFIED]")
        # Remove discharge summary/diagnosis sections (label leakage)
        t = re.sub(
            r"(?:DISCHARGE\s+DIAGNOSIS|FINAL\s+DIAGNOSIS|DISCHARGE\s+SUMMARY|ASSESSMENT/PLAN|A/P)[:\-\n].*",
            "",
            t,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return " ".join(t.split())

    def _lab_text(self, hadm_id: int) -> str:
        df = self.labs_df[self.labs_df["hadm_id"] == hadm_id].copy()
        if df.empty:
            return ""
        df["label"] = df["label"].astype(str).str.lower()
        parts = []
        for name, pattern in self._LAB_KEYS:
            sub = df[df["label"].str.contains(pattern, regex=True, na=False)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            val = "" if pd.isna(r.get("valuenum")) else str(r.get("valuenum"))
            uom = "" if pd.isna(r.get("valueuom")) else f" {r.get('valueuom')}"
            if not val:
                raw = r.get("value")
                val = "" if pd.isna(raw) else str(raw)
            parts.append(f"{name}: {val}{uom}".strip())
        return ("Laboratory Results: " + "; ".join(parts)) if parts else ""

    def _build_clinical_text(self, row: pd.Series) -> str:
        evidence_parts = []
        lab_text = self._lab_text(int(row["hadm_id"]))
        if lab_text:
            evidence_parts.append(lab_text)

        for col, prefix in [
            ("ECG", "ECG:"),
            ("reports", "ECG Reports:"),
            ("Ultrasound", "Echo:"),
            ("X-ray", "Chest X-ray:"),
            ("CT", "CT:"),
            ("MRI", "MRI:"),
        ]:
            if col in self.diagnoses_df.columns and pd.notna(row.get(col)):
                txt = self._clean(row[col])
                if len(txt) > 10:
                    evidence_parts.append(f"{prefix} {txt}")

        parts = []
        if evidence_parts:
            parts.append("[EVIDENCE] " + " ".join(evidence_parts))

        for col, head in [
            ("chief_complaint", "Chief Complaint:"),
            ("HPI", "History:"),
            ("physical_exam", "Physical Exam:"),
        ]:
            if col in self.diagnoses_df.columns and pd.notna(row.get(col)):
                txt = self._clean(row[col])
                if txt:
                    parts.append(f"[CONTEXT] {head} {txt}")

        text = " ".join(parts).strip()
        if len(text) < 50:
            return "[EVIDENCE] No diagnostic data. [CONTEXT] Minimal clinical information."
        return text

    # ── Splits ────────────────────────────────────────────────────────────────

    def prepare_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Building clinical text...")
        self.diagnoses_df["clinical_text"] = self.diagnoses_df.apply(
            self._build_clinical_text, axis=1
        )
        self.diagnoses_df = self.diagnoses_df[
            self.diagnoses_df["clinical_text"].str.len() > 100
        ].copy()
        print(f"  Retained: {len(self.diagnoses_df)} admissions")

        # Subject-level split to prevent data leakage
        subj_label = self.diagnoses_df.groupby("subject_id")["label"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        subjects = subj_label.index.to_numpy()
        labels = subj_label.values

        tr_sub, tmp_sub, _, tmp_y = train_test_split(
            subjects, labels, test_size=0.30, random_state=SEED, stratify=labels
        )
        va_sub, te_sub = train_test_split(
            tmp_sub, test_size=0.50, random_state=SEED, stratify=tmp_y
        )

        def _subset(subs):
            return self.diagnoses_df[
                self.diagnoses_df["subject_id"].isin(subs)
            ].reset_index(drop=True)

        train_df, val_df, test_df = _subset(tr_sub), _subset(va_sub), _subset(te_sub)
        print(
            f"  Split — train={len(train_df)} | val={len(val_df)} | test={len(test_df)}"
        )
        return train_df, val_df, test_df

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "MIMICCardiacDataProcessor":
        with open(path, "rb") as f:
            return pickle.load(f)


class ClinicalTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 4096, special_token_ids=None):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.special_ids = special_token_ids or []

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        enc = self.tok(
            row["clinical_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        gmask = torch.zeros_like(enc["attention_mask"])
        gmask[:, 0] = 1  # global attention on CLS
        for tid in self.special_ids:
            gmask |= (enc["input_ids"] == tid).long()

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "global_attention_mask": gmask.squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
            "hadm_id": int(row["hadm_id"]),
        }
