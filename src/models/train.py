"""
Training loop for Clinical-Longformer cardiovascular classifier.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LongformerForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from config import (
    BATCH_SIZE,
    EPOCHS,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_NAME,
    SEED,
    SPECIAL_TOKENS,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
)
from src.data.processor import ClinicalTextDataset
from src.utils.helpers import set_seed


def _train_epoch(model, loader, optimizer, criterion, device, scheduler, scaler, grad_accum):
    model.train()
    total_loss = 0.0
    accum_loss = 0.0

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        gm = batch["global_attention_mask"].to(device)
        y = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model(input_ids=ids, attention_mask=am, global_attention_mask=gm, labels=y)
            loss = criterion(out.logits, y) / grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            total_loss += accum_loss * grad_accum
            accum_loss = 0.0

    # handle remaining accumulation
    if accum_loss > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        total_loss += accum_loss * grad_accum

    return total_loss / max(1, len(loader))


def _evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            gm = batch["global_attention_mask"].to(device)
            y = batch["labels"].to(device)

            logits = model(input_ids=ids, attention_mask=am, global_attention_mask=gm).logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    probs_np = np.vstack(all_probs) if all_probs else np.zeros((0,))
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return acc, all_preds, all_labels, probs_np


def train(train_df, val_df, test_df, label_map: dict, output_dir: str):
    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)

    num_labels = len(label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Class-balanced loss weights
    counts = train_df["label"].value_counts().reindex(range(num_labels), fill_value=0)
    w = np.array(
        [len(train_df) / (num_labels * max(1, c)) for c in counts.values], dtype=np.float32
    )
    w = np.minimum(w, w.mean() * 5.0)
    class_weights = torch.tensor(w, dtype=torch.float32).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = LongformerForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.to(device)

    sp_ids = tokenizer.additional_special_tokens_ids
    train_ds = ClinicalTextDataset(train_df, tokenizer, MAX_LENGTH, sp_ids)
    val_ds = ClinicalTextDataset(val_df, tokenizer, MAX_LENGTH, sp_ids)
    test_ds = ClinicalTextDataset(test_df, tokenizer, MAX_LENGTH, sp_ids)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    total_steps = (len(train_loader) * EPOCHS) // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_FRACTION * total_steps), total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_f1, history = -1.0, []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        tr_loss = _train_epoch(model, train_loader, optimizer, criterion, device, scheduler, scaler, GRAD_ACCUM_STEPS)
        val_acc, val_pred, val_true, _ = _evaluate(model, val_loader, device)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        print(f"  loss={tr_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "val_acc": val_acc, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            print("  New best — saving model")
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "val_f1": val_f1},
                os.path.join(output_dir, "best_model.pt"),
            )
            save_dir = os.path.join(output_dir, "final_model")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    # Load best and run test
    ckpt = torch.load(os.path.join(output_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print("\nExporting predictions...")
    import pandas as pd

    for split, df_split, loader in [("val", val_df, val_loader), ("test", test_df, test_loader)]:
        acc, preds, labels, probs = _evaluate(model, loader, device)
        out = pd.DataFrame(probs, columns=[f"prob_class_{i}" for i in range(num_labels)])
        out["true_label"] = labels
        out["predicted_label"] = preds
        out["hadm_id"] = df_split["hadm_id"].values
        out["clinical_text"] = df_split["clinical_text"].values
        out.to_csv(os.path.join(output_dir, f"{split}_prob_matrix.csv"), index=False)
        np.save(os.path.join(output_dir, f"{split}_probs.npy"), probs)
        np.save(os.path.join(output_dir, f"{split}_labels.npy"), np.array(labels))

    te_acc, te_pred, te_true, _ = _evaluate(model, test_loader, device)
    classes = [label_map[i] for i in range(num_labels)]
    report = classification_report(te_true, te_pred, target_names=classes, digits=3, zero_division=0)
    print("\n" + report)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(te_true, te_pred)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        os.path.join(output_dir, "confusion_matrix.csv")
    )
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump({"label_to_name": label_map, "name_to_label": {v: k for k, v in label_map.items()}}, f, indent=2)

    print(f"Test accuracy: {te_acc:.3f}")
    return model, tokenizer
