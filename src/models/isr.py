"""
Information Sufficiency Ratio (ISR) gating for selective prediction.

Two gating modes:
  - ISR-only : abstain when ISR < threshold (calibrated on validation to meet error target h*)
  - Hybrid   : accept if (ISR ≥ threshold) OR (calibrated confidence ≥ τ AND margin ≥ γ)
"""

import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import B_CLIP, CONTEXT_HEADERS, EVIDENCE_HEADERS, H_STAR_HYBRID, H_STAR_ISR, M_PERM, MAX_LENGTH, PRIOR_MODE, SEED
from src.utils.helpers import set_seed, wilson_bounds


# ── Section permutation ───────────────────────────────────────────────────────

def _extract_sections(text: str, headers: list[str]) -> list[tuple[str, str]]:
    if not isinstance(text, str) or not text.strip():
        return []
    pattern = re.compile("|".join(re.escape(h) for h in headers))
    matches = list(pattern.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[m.end() : end].strip()
        if content:
            sections.append((m.group(0), content))
    return sections


def _shuffle_non_identity(items: list) -> list:
    if len(items) <= 1:
        return items[:]
    original = items[:]
    for _ in range(10):
        candidate = items[:]
        random.shuffle(candidate)
        if candidate != original:
            return candidate
    return items[:]


def permute_sections(text: str, seed: int | None = None) -> str:
    """Return a copy of *text* with clinical sections reordered."""
    if seed is not None:
        random.seed(seed)
    if "[EVIDENCE]" not in text or "[CONTEXT]" not in text:
        return text

    evidence_raw = text.split("[EVIDENCE]", 1)[1].split("[CONTEXT]", 1)[0].strip()
    context_raw = text.split("[CONTEXT]", 1)[1].strip()

    ev_sections = _extract_sections(evidence_raw, EVIDENCE_HEADERS)
    ctx_sections = _extract_sections(context_raw, CONTEXT_HEADERS)

    ev_sections = _shuffle_non_identity(ev_sections)
    ctx_sections = _shuffle_non_identity(ctx_sections)

    parts = []
    if ev_sections:
        parts.append("[EVIDENCE] " + " ".join(f"{h} {c}" for h, c in ev_sections))
    if ctx_sections:
        parts.append("[CONTEXT] " + " ".join(f"{h} {c}" for h, c in ctx_sections))
    return " ".join(parts).strip() or text


# ── ISR Gating ────────────────────────────────────────────────────────────────

class ISRGating:
    """
    Post-hoc ISR gating over a trained Clinical-Longformer model.

    Parameters
    ----------
    model_path   : directory containing ``final_model/`` (HuggingFace format)
    val_csv      : path to validation prob-matrix CSV (output of training)
    test_csv     : path to test prob-matrix CSV
    h_star_isr   : ISR-only target Wilson upper error bound
    h_star_hybrid: Hybrid target Wilson upper error bound
    B            : delta clip bound (nats)
    m            : number of section permutations
    max_len      : tokeniser truncation length
    prior_mode   : how to compute q̂ — one of {q25, q10, mean, min}
    """

    def __init__(
        self,
        model_path: str,
        val_csv: str,
        test_csv: str,
        h_star_isr: float = H_STAR_ISR,
        h_star_hybrid: float = H_STAR_HYBRID,
        B: float = B_CLIP,
        m: int = M_PERM,
        max_len: int = MAX_LENGTH,
        prior_mode: str = PRIOR_MODE,
    ):
        self.model_path = model_path
        self.val = pd.read_csv(val_csv)
        self.test = pd.read_csv(test_csv)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from {model_path}/final_model  (device={self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/final_model")
        self.tokenizer.model_max_length = 10**7
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(f"{model_path}/final_model")
            .to(self.device)
            .eval()
        )

        self.h_star_isr = float(h_star_isr)
        self.h_star_hybrid = float(h_star_hybrid)
        self.B = float(B)
        self.m = int(m)
        self.max_len = int(max_len)
        self.prior_mode = prior_mode
        self.temperature = 1.0

        self._cache_raw: dict = {}
        self._cache_cal: dict = {}

        set_seed(SEED)
        self._calibrate_temperature()

    # ── Temperature calibration ───────────────────────────────────────────────

    def _calibrate_temperature(self, n_max: int = 800) -> None:
        print("Calibrating temperature on validation set...")
        logits_list, y_list = [], []
        n = min(n_max, len(self.val))

        with torch.no_grad():
            for i in range(n):
                if i % 200 == 0:
                    print(f"  {i}/{n}")
                row = self.val.iloc[i]
                logits_list.append(self._forward_logits(row["clinical_text"]))
                y_list.append(int(row["true_label"]))

        logits = torch.stack(logits_list)
        labels = torch.tensor(y_list)

        T = nn.Parameter(torch.ones(()))
        opt = torch.optim.LBFGS([T], lr=0.5, max_iter=60, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / T.clamp_min(0.1), labels)
            loss.backward()
            return loss

        for _ in range(10):
            opt.step(closure)

        with torch.no_grad():
            self.temperature = float(T.clamp(0.1, 10.0).item())
            nll = F.cross_entropy(logits / self.temperature, labels).item()
        print(f"  Optimal T={self.temperature:.3f}  (val NLL={nll:.4f})")

    # ── Forward passes ────────────────────────────────────────────────────────

    def _forward_logits(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_len, padding=True
        ).to(self.device)
        gmask = torch.zeros_like(enc["attention_mask"])
        gmask[:, 0] = 1
        return self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            global_attention_mask=gmask,
        ).logits[0].detach().cpu()

    def _probs_raw(self, text: str) -> np.ndarray:
        key = (hash(text), self.max_len, "raw")
        if key not in self._cache_raw:
            with torch.no_grad():
                self._cache_raw[key] = torch.softmax(self._forward_logits(text), dim=-1).numpy()
        return self._cache_raw[key]

    def _probs_cal(self, text: str) -> np.ndarray:
        key = (hash(text), self.max_len, round(self.temperature, 3), "cal")
        if key not in self._cache_cal:
            with torch.no_grad():
                logits = self._forward_logits(text) / self.temperature
                self._cache_cal[key] = torch.softmax(logits, dim=-1).numpy()
        return self._cache_cal[key]

    # ── Prior estimation ──────────────────────────────────────────────────────

    def _prior(self, vals: list[float]) -> float:
        s = np.clip(np.asarray(vals, dtype=float), 1e-8, 1 - 1e-8)
        if self.prior_mode == "min":
            return float(s.min())
        if self.prior_mode == "mean":
            return float(s.mean())
        if self.prior_mode == "q10":
            return float(np.quantile(s, 0.10))
        return float(np.quantile(s, 0.25))  # default: q25

    # ── ISR computation ───────────────────────────────────────────────────────

    def compute_isr(self, text: str, true_label=None) -> dict:
        p_raw = self._probs_raw(text)
        y_pred = int(p_raw.argmax())
        p_full = float(p_raw[y_pred])

        p_cal = self._probs_cal(text)
        p_full_cal = float(p_cal[y_pred])
        top2 = np.sort(p_cal)[-2:]
        margin_cal = float(top2[-1] - top2[-2])

        perm_probs, deltas = [], []
        for i in range(self.m):
            perm_text = permute_sections(text, seed=i)
            p_perm = float(self._probs_raw(perm_text)[y_pred])
            perm_probs.append(p_perm)
            delta = np.log(p_full + 1e-12) - np.log(p_perm + 1e-12)
            deltas.append(float(np.clip(delta, 0.0, self.B)))

        q_hat = self._prior(perm_probs)
        d = np.sort(np.asarray(deltas, dtype=float))
        if len(d) >= 5:
            d = d[1:-1]  # trim extremes
        delta_bar = float(d.mean()) if len(d) else 0.0

        p_target = 1.0 - self.h_star_isr
        b2t = p_target * np.log(p_target / q_hat) + (1 - p_target) * np.log(
            (1 - p_target) / (1 - q_hat)
        )
        isr = delta_bar / b2t if b2t > 0 else 0.0

        return {
            "predicted_class": y_pred,
            "true_label": int(true_label) if true_label is not None else None,
            "correct": (y_pred == int(true_label)) if true_label is not None else None,
            "p_full_raw": p_full,
            "p_full_cal": p_full_cal,
            "margin_cal": margin_cal,
            "q_hat": float(q_hat),
            "delta_bar": float(delta_bar),
            "b2t": float(b2t),
            "isr": float(isr),
            "perm_std": float(np.std(perm_probs)),
        }

    # ── Table computation ─────────────────────────────────────────────────────

    def compute_table(self, split: str = "val") -> pd.DataFrame:
        df_src = self.val if split == "val" else self.test
        self._cache_raw.clear()
        self._cache_cal.clear()

        rows = []
        print(f"\nComputing ISR on {split} set ({len(df_src)} samples)...")
        for i in range(len(df_src)):
            if i % 100 == 0:
                print(f"  {i}/{len(df_src)}")
            r = self.compute_isr(df_src.iloc[i]["clinical_text"], int(df_src.iloc[i]["true_label"]))
            rows.append(r)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.model_path, f"{split}_isr_table.csv"), index=False)
        return df

    # ── Threshold selection ───────────────────────────────────────────────────

    def select_isr_threshold(self, df_val: pd.DataFrame, h_target: float | None = None) -> dict | None:
        h_target = h_target or self.h_star_isr
        best = None
        for eps in np.linspace(0.0, 1.2, 61):
            thr = 1.0 + eps
            sub = df_val[df_val["isr"] >= thr]
            if len(sub) == 0:
                continue
            err = (~sub["correct"]).mean()
            _, wu = wilson_bounds(err, len(sub))
            if wu <= h_target:
                cfg = {
                    "threshold": float(thr),
                    "coverage_val": float(len(sub) / len(df_val)),
                    "err_val": float(err),
                    "wilson_upper_val": float(wu),
                }
                if best is None or cfg["coverage_val"] > best["coverage_val"]:
                    best = cfg
        if best:
            print("ISR threshold:", json.dumps(best, indent=2))
        else:
            print(f"No ISR threshold met h*={h_target:.0%} on validation")
        return best

    def select_hybrid_params(self, df_val: pd.DataFrame, isr_cfg: dict, h_target: float | None = None) -> dict | None:
        h_target = h_target or self.h_star_hybrid
        thr = isr_cfg["threshold"]
        best = None
        for tau in np.linspace(0.70, 0.98, 29):
            for gamma in np.linspace(0.00, 0.20, 11):
                answered = (df_val["isr"] >= thr) | (
                    (df_val["p_full_cal"] >= tau) & (df_val["margin_cal"] >= gamma)
                )
                sub = df_val[answered]
                if len(sub) == 0:
                    continue
                err = (~sub["correct"]).mean()
                _, wu = wilson_bounds(err, len(sub))
                if wu <= h_target:
                    cfg = {
                        "threshold": float(thr),
                        "tau": float(tau),
                        "gamma": float(gamma),
                        "coverage_val": float(len(sub) / len(df_val)),
                        "err_val": float(err),
                        "wilson_upper_val": float(wu),
                    }
                    if best is None or cfg["coverage_val"] > best["coverage_val"]:
                        best = cfg
        if best:
            print("Hybrid params:", json.dumps(best, indent=2))
        else:
            print(f"No hybrid params met h*={h_target:.0%} on validation")
        return best

    # ── Evaluation ────────────────────────────────────────────────────────────

    @staticmethod
    def _summarize(df: pd.DataFrame, mask=None, label: str = "") -> dict:
        if mask is None:
            acc = float(df["correct"].mean())
            return {"label": label, "coverage": 1.0, "accuracy": acc, "n": len(df)}
        sub = df[mask]
        cov = len(sub) / len(df) if len(df) else 0.0
        if len(sub) == 0:
            return {"label": label, "coverage": float(cov), "accuracy": None, "n": 0}
        acc = float(sub["correct"].mean())
        err = 1.0 - acc
        wl, wu = wilson_bounds(err, len(sub))
        return {
            "label": label,
            "coverage": float(cov),
            "accuracy": float(acc),
            "error": float(err),
            "wilson_lower": float(wl),
            "wilson_upper": float(wu),
            "n": int(len(sub)),
        }

    def evaluate(self, df_test: pd.DataFrame, isr_cfg: dict, hybrid_cfg: dict | None = None) -> dict:
        results = {
            "baseline_ungated": self._summarize(df_test, None, "baseline_ungated"),
            "isr_only": self._summarize(df_test, df_test["isr"] >= isr_cfg["threshold"], "isr_only"),
        }
        if hybrid_cfg is not None:
            hyb_mask = (df_test["isr"] >= hybrid_cfg["threshold"]) | (
                (df_test["p_full_cal"] >= hybrid_cfg["tau"])
                & (df_test["margin_cal"] >= hybrid_cfg["gamma"])
            )
            results["hybrid"] = self._summarize(df_test, hyb_mask, "hybrid")
        return results

    # ── Coverage–accuracy curve ───────────────────────────────────────────────

    @staticmethod
    def coverage_curve(df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
        if thresholds is None:
            thresholds = np.linspace(0.5, 5.0, 91)
        rows = []
        for thr in thresholds:
            sub = df[df["isr"] >= thr]
            if len(sub) == 0:
                continue
            rows.append({
                "threshold": float(thr),
                "coverage": float(len(sub) / len(df)),
                "accuracy": float(sub["correct"].mean()),
                "n": int(len(sub)),
            })
        return pd.DataFrame(rows).sort_values("coverage", ascending=False)
