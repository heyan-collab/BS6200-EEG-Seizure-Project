#!/usr/bin/env python
"""
BS6200 CNN-BiLSTM V3 pipeline using a FIXED shared train/test split and FIXED CV folds.

Designed for:
- UCI/Kaggle Epileptic Seizure Recognition data
- data columns: X1..X178 (case-insensitive) and y
- shared split CSV columns: row_index, split, cv_fold
- split labels: development / final_test
- binary task: y==1 seizure, y in {2,3,4,5} non-seizure
- five-class task: original labels 1..5 mapped internally to 0..4

Important:
- No random train/test split is generated.
- No random CV folds are generated.
- Scaling is fit only on each training fold.
- Final test set is untouched until final evaluation.
- Every run gets a timestamped output directory, so older results are not overwritten.
"""

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 6200


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ============================================================
# Data loading
# ============================================================

def detect_feature_columns(df):
    """Find X1..X178 columns case-insensitively and return them in numeric order."""
    matched = []
    for c in df.columns:
        m = re.fullmatch(r"[xX](\d+)", str(c).strip())
        if m:
            matched.append((int(m.group(1)), c))

    matched.sort(key=lambda z: z[0])
    feature_cols = [c for _, c in matched]

    if len(feature_cols) != 178:
        raise ValueError(
            f"Expected 178 EEG feature columns X1..X178, but found {len(feature_cols)}."
        )
    return feature_cols


def detect_label_column(df):
    for c in df.columns:
        if str(c).strip().lower() == "y":
            return c
    raise ValueError("Could not find label column 'y'.")


def load_dataset(data_path):
    df = pd.read_csv(data_path)
    feature_cols = detect_feature_columns(df)
    label_col = detect_label_column(df)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_original = df[label_col].to_numpy(dtype=int)

    if np.isnan(X).any():
        raise ValueError("NaN values found in EEG features. Please inspect the dataset first.")

    expected_labels = {1, 2, 3, 4, 5}
    labels_found = set(np.unique(y_original).tolist())
    if not labels_found.issubset(expected_labels):
        raise ValueError(
            f"Unexpected labels found in y: {sorted(labels_found)}. Expected labels 1..5."
        )

    print(f"Data shape: {df.shape}")
    print(f"X shape: {X.shape}")
    print("Original class counts:")
    print(pd.Series(y_original).value_counts().sort_index().to_string())

    return df, X, y_original


def load_shared_split(split_path, n_samples):
    split_df = pd.read_csv(split_path)

    required = {"row_index", "split", "cv_fold"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(
            f"Split file is missing required columns: {sorted(missing)}"
        )

    if len(split_df) != n_samples:
        raise ValueError(
            f"Split file has {len(split_df)} rows, but dataset has {n_samples} rows."
        )

    if split_df["row_index"].duplicated().any():
        raise ValueError("Duplicate row_index values found in split file.")

    split_df = split_df.sort_values("row_index").reset_index(drop=True)

    expected_index = np.arange(n_samples)
    actual_index = split_df["row_index"].to_numpy(dtype=int)

    if not np.array_equal(actual_index, expected_index):
        raise ValueError(
            "row_index in split file must be exactly 0..N-1 and match dataset row order."
        )

    allowed_splits = {"development", "final_test"}
    actual_splits = set(split_df["split"].dropna().astype(str).unique().tolist())
    if not actual_splits.issubset(allowed_splits):
        raise ValueError(
            f"Unexpected split labels: {sorted(actual_splits)}. "
            "Expected 'development' and 'final_test'."
        )

    dev_mask = split_df["split"].astype(str).eq("development")
    test_mask = split_df["split"].astype(str).eq("final_test")

    dev_idx = split_df.loc[dev_mask, "row_index"].to_numpy(dtype=int)
    test_idx = split_df.loc[test_mask, "row_index"].to_numpy(dtype=int)

    if len(dev_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Development or final_test split is empty.")

    if split_df.loc[dev_mask, "cv_fold"].isna().any():
        raise ValueError("Some development samples have missing cv_fold values.")

    fold_ids = sorted(
        split_df.loc[dev_mask, "cv_fold"].astype(int).unique().tolist()
    )

    # Strong check for the agreed BS6200 protocol
    if fold_ids != [0, 1, 2, 3, 4]:
        raise ValueError(
            f"Expected fixed CV folds [0,1,2,3,4], but found {fold_ids}."
        )

    # Final-test rows should not participate in CV.
    # NaN is preferred, but numeric values are simply ignored because split controls membership.
    print(f"Development samples: {len(dev_idx)}")
    print(f"Final test samples: {len(test_idx)}")
    print(f"Fixed CV folds: {fold_ids}")

    for fold in fold_ids:
        n_val = int(
            (
                dev_mask
                & split_df["cv_fold"].astype("Int64").eq(fold)
            ).sum()
        )
        print(f"  Fold {fold}: validation={n_val}, training={len(dev_idx)-n_val}")

    return split_df, dev_idx, test_idx, fold_ids


# ============================================================
# Labels
# ============================================================

def make_labels(y_original, task):
    if task == "binary":
        # seizure = 1; non-seizure = 0
        return (y_original == 1).astype(np.int64), 2
    elif task == "five":
        # original 1..5 -> internal 0..4
        return (y_original - 1).astype(np.int64), 5
    else:
        raise ValueError(task)


# ============================================================
# Model
# ============================================================

class CNNBiLSTM(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: [batch, 178]
        x = x.unsqueeze(1)       # [batch, 1, 178]
        x = self.cnn(x)          # [batch, channels, time]
        x = x.transpose(1, 2)    # [batch, time, channels]
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        return self.classifier(x)


# ============================================================
# Training helpers
# ============================================================

def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,   # safest for Windows / Anaconda / PyCharm
        pin_memory=torch.cuda.is_available(),
    )


def get_class_weights(y_train, n_classes, device):
    classes = np.arange(n_classes)
    present = np.unique(y_train)

    if len(present) != n_classes:
        raise ValueError(
            f"Training fold is missing one or more classes. Present classes: {present.tolist()}"
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * len(yb)
            total_n += len(yb)

    return total_loss / max(total_n, 1)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    max_epochs,
    patience,
):
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(yb)
            train_n += len(yb)

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"    epoch {epoch:02d}/{max_epochs} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"    Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    return pd.DataFrame(history), best_epoch, best_val_loss


def train_fixed_epochs(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs,
):
    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(yb)
            train_n += len(yb)

        train_loss = train_loss_sum / max(train_n, 1)
        history.append({"epoch": epoch, "train_loss": train_loss})

        print(
            f"    final-train epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.5f}"
        )

    return pd.DataFrame(history)


def predict(model, X, batch_size, device):
    dummy_y = np.zeros(len(X), dtype=np.int64)
    loader = make_loader(X, dummy_y, batch_size=batch_size, shuffle=False)

    model.eval()
    all_logits = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)

    # stable softmax
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)

    pred = probs.argmax(axis=1)
    return pred, probs


# ============================================================
# Metrics
# ============================================================

def binary_metrics(y_true, y_pred, probs):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs[:, 1]),
        "pr_auc": average_precision_score(y_true, probs[:, 1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return out


def five_class_metrics(y_true, y_pred, probs):
    labels = np.arange(5)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }

    try:
        y_onehot = np.eye(5, dtype=np.float32)[y_true]
        out["macro_roc_auc_ovr"] = roc_auc_score(
            y_onehot, probs, average="macro", multi_class="ovr"
        )
    except Exception:
        out["macro_roc_auc_ovr"] = np.nan

    return out


def calculate_metrics(task, y_true, y_pred, probs):
    if task == "binary":
        return binary_metrics(y_true, y_pred, probs)
    return five_class_metrics(y_true, y_pred, probs)


# ============================================================
# Output helpers
# ============================================================

def save_confusion_matrix(y_true, y_pred, task, path):
    if task == "binary":
        labels = [0, 1]
        display_labels = ["Non-seizure", "Seizure"]
    else:
        labels = [0, 1, 2, 3, 4]
        display_labels = ["1", "2", "3", "4", "5"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display_labels)
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{task.capitalize()} confusion matrix")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curve(history_df, path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")

    if "val_loss" in history_df.columns:
        ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    return obj


def summarize_cv(cv_df, metric_cols):
    summary = {}
    for c in metric_cols:
        vals = pd.to_numeric(cv_df[c], errors="coerce")
        summary[c] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
        }
    return summary


# ============================================================
# One task
# ============================================================

def run_task(
    task,
    X,
    y_original,
    split_df,
    dev_idx,
    test_idx,
    fold_ids,
    task_dir,
    args,
    device,
):
    print("\n" + "=" * 70)
    print(f"TASK: {task}")
    print("=" * 70)

    y, n_classes = make_labels(y_original, task)

    dev_mask = split_df["split"].astype(str).eq("development")
    dev_split_df = split_df.loc[dev_mask].copy()

    cv_rows = []
    best_epochs = []

    for fold in fold_ids:
        print(f"\n--- {task}: fixed CV fold {fold} ---")

        val_idx = dev_split_df.loc[
            dev_split_df["cv_fold"].astype(int).eq(fold),
            "row_index",
        ].to_numpy(dtype=int)

        train_idx = dev_split_df.loc[
            ~dev_split_df["cv_fold"].astype(int).eq(fold),
            "row_index",
        ].to_numpy(dtype=int)

        print(f"train={len(train_idx)}, validation={len(val_idx)}")

        # Leakage-free scaling: fit ONLY on training fold.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
        X_val = scaler.transform(X[val_idx]).astype(np.float32)

        y_train = y[train_idx]
        y_val = y[val_idx]

        train_loader = make_loader(
            X_train, y_train, batch_size=args.batch_size, shuffle=True
        )
        val_loader = make_loader(
            X_val, y_val, batch_size=args.batch_size, shuffle=False
        )

        set_seed(args.seed + fold)
        model = CNNBiLSTM(n_classes=n_classes).to(device)

        if args.no_class_weights:
            criterion = nn.CrossEntropyLoss()
        else:
            class_weights = get_class_weights(y_train, n_classes, device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        history_df, best_epoch, best_val_loss = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_epochs=args.epochs,
            patience=args.patience,
        )

        best_epochs.append(best_epoch)

        fold_dir = task_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        history_df.to_csv(fold_dir / "training_history.csv", index=False)
        save_training_curve(
            history_df,
            fold_dir / "training_history.png",
            f"{task} - fold {fold}",
        )

        y_pred, probs = predict(
            model, X_val, batch_size=args.batch_size, device=device
        )
        metrics = calculate_metrics(task, y_val, y_pred, probs)
        metrics["fold"] = int(fold)
        metrics["best_epoch"] = int(best_epoch)
        metrics["best_val_loss"] = float(best_val_loss)
        cv_rows.append(metrics)

        pred_df = pd.DataFrame(
            {
                "row_index": val_idx,
                "true_internal": y_val,
                "pred_internal": y_pred,
            }
        )

        if task == "binary":
            pred_df["true_original"] = np.where(y_val == 1, 1, 0)
            pred_df["prob_seizure"] = probs[:, 1]
        else:
            pred_df["true_original"] = y_val + 1
            pred_df["pred_original"] = y_pred + 1
            for k in range(5):
                pred_df[f"prob_class_{k+1}"] = probs[:, k]

        pred_df.to_csv(fold_dir / "validation_predictions.csv", index=False)

        save_confusion_matrix(
            y_val,
            y_pred,
            task,
            fold_dir / "confusion_matrix.png",
        )

        print("Fold metrics:")
        for k, v in metrics.items():
            if k != "fold":
                print(f"  {k}: {v}")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(task_dir / "cv_fold_metrics.csv", index=False)

    metric_cols = [
        c for c in cv_df.columns
        if c not in {"fold", "best_epoch", "best_val_loss", "tn", "fp", "fn", "tp"}
    ]
    cv_summary = summarize_cv(cv_df, metric_cols)

    # Epoch count for final model is selected ONLY from CV, never from test.
    final_epochs = int(np.median(best_epochs))
    final_epochs = max(final_epochs, 1)

    print(f"\nCV best epochs: {best_epochs}")
    print(f"Final retraining epochs (median CV best epoch): {final_epochs}")

    # Fit scaler on ALL development samples only.
    final_scaler = StandardScaler()
    X_dev = final_scaler.fit_transform(X[dev_idx]).astype(np.float32)
    X_test = final_scaler.transform(X[test_idx]).astype(np.float32)

    y_dev = y[dev_idx]
    y_test = y[test_idx]

    dev_loader = make_loader(
        X_dev, y_dev, batch_size=args.batch_size, shuffle=True
    )

    set_seed(args.seed + 100)
    final_model = CNNBiLSTM(n_classes=n_classes).to(device)

    if args.no_class_weights:
        final_criterion = nn.CrossEntropyLoss()
    else:
        final_weights = get_class_weights(y_dev, n_classes, device)
        final_criterion = nn.CrossEntropyLoss(weight=final_weights)

    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr)

    final_history = train_fixed_epochs(
        model=final_model,
        train_loader=dev_loader,
        criterion=final_criterion,
        optimizer=final_optimizer,
        device=device,
        epochs=final_epochs,
    )

    final_history.to_csv(task_dir / "final_training_history.csv", index=False)
    save_training_curve(
        final_history,
        task_dir / "final_training_history.png",
        f"{task} - final model",
    )

    y_test_pred, test_probs = predict(
        final_model, X_test, batch_size=args.batch_size, device=device
    )

    test_metrics = calculate_metrics(task, y_test, y_test_pred, test_probs)

    print("\nFINAL TEST METRICS")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    save_confusion_matrix(
        y_test,
        y_test_pred,
        task,
        task_dir / "final_test_confusion_matrix.png",
    )

    test_pred_df = pd.DataFrame(
        {
            "row_index": test_idx,
            "true_internal": y_test,
            "pred_internal": y_test_pred,
        }
    )

    if task == "binary":
        test_pred_df["true_original"] = np.where(y_test == 1, 1, 0)
        test_pred_df["prob_seizure"] = test_probs[:, 1]
    else:
        test_pred_df["true_original"] = y_test + 1
        test_pred_df["pred_original"] = y_test_pred + 1
        for k in range(5):
            test_pred_df[f"prob_class_{k+1}"] = test_probs[:, k]

    test_pred_df.to_csv(task_dir / "final_test_predictions.csv", index=False)

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "n_classes": n_classes,
            "task": task,
            "final_epochs": final_epochs,
            "scaler_mean": final_scaler.mean_,
            "scaler_scale": final_scaler.scale_,
        },
        task_dir / "final_model.pt",
    )

    result_summary = {
        "task": task,
        "shared_split_file": str(args.split),
        "development_n": int(len(dev_idx)),
        "final_test_n": int(len(test_idx)),
        "cv_folds": [int(f) for f in fold_ids],
        "maximum_epochs_per_cv_fold": int(args.epochs),
        "cv_best_epochs": [int(x) for x in best_epochs],
        "final_retraining_epochs": int(final_epochs),
        "cv_summary": cv_summary,
        "final_test_metrics": test_metrics,
    }

    with open(task_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(result_summary), f, indent=2, ensure_ascii=False)

    return result_summary


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "BS6200 CNN-BiLSTM using the shared fixed train/test split "
            "and fixed 5-fold CV assignment."
        )
    )

    parser.add_argument(
        "--data",
        default="data(1).csv",
        help="Path to UCI/Kaggle EEG CSV.",
    )
    parser.add_argument(
        "--split",
        default="splits/shared_split_with_cv.csv",
        help="Path to shared split CSV containing row_index, split, cv_fold. "
             "Default: splits/shared_split_with_cv.csv",
    )
    parser.add_argument(
        "--output",
        default="results_cnn_bilstm_sharedsplit",
        help="Parent output directory. Each run gets its own timestamped folder.",
    )
    parser.add_argument(
        "--task",
        choices=["binary", "five", "both"],
        default="both",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum epochs per CV fold. Early stopping may stop earlier.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience for CV folds.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable balanced class weights.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test using fixed folds 0 and 1, with max 3 epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve relative paths from the folder containing this script.
    # This matches the user's project layout:
    #
    # project_root/
    #   cnn_bilstm_sharedsplit_v3.py
    #   data(1).csv
    #   splits/
    #       shared_split_with_cv.csv
    #
    # Therefore the script works even if Anaconda Prompt was opened elsewhere.
    script_dir = Path(__file__).resolve().parent

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = script_dir / data_path

    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = script_dir / split_path

    output_parent = Path(args.output)
    if not output_parent.is_absolute():
        output_parent = script_dir / output_parent

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Expected the dataset in the project root, or pass --data with another path."
        )
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            "Expected: splits/shared_split_with_cv.csv, "
            "or pass --split with the actual file path."
        )

    df, X, y_original = load_dataset(data_path)
    split_df, dev_idx, test_idx, fold_ids = load_shared_split(
        split_path, n_samples=len(df)
    )

    # Quick mode still respects the shared fixed fold assignments.
    if args.quick:
        fold_ids = [0, 1]
        args.epochs = min(args.epochs, 3)
        run_prefix = "quick"
    else:
        run_prefix = f"full_{args.epochs}epochs"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_parent / f"{run_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Data file: {data_path}")
    print(f"Shared split file: {split_path}")
    print(f"Output parent: {output_parent}")
    print(f"Task: {args.task}")
    print(f"Maximum epochs per CV fold: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Patience: {args.patience}")

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "run_directory": str(run_dir),
            "development_n": int(len(dev_idx)),
            "final_test_n": int(len(test_idx)),
            "fold_ids_used": [int(f) for f in fold_ids],
        }
    )

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, indent=2, ensure_ascii=False)

    tasks = ["binary", "five"] if args.task == "both" else [args.task]

    all_results = {}

    for task in tasks:
        task_dir = run_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        all_results[task] = run_task(
            task=task,
            X=X,
            y_original=y_original,
            split_df=split_df,
            dev_idx=dev_idx,
            test_idx=test_idx,
            fold_ids=fold_ids,
            task_dir=task_dir,
            args=args,
            device=device,
        )

    with open(run_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Results saved to: {run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
