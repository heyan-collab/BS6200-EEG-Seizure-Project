"""
BS6200 EEG Project - CNN + BiLSTM Reference Pipeline
====================================================

Dataset expected:
    column_a, x1, x2, ..., x178, y

Tasks:
    1) Binary seizure detection:
         y=1 -> seizure (1)
         y=2,3,4,5 -> non-seizure (0)
    2) Five-class EEG state classification:
         original y=1..5 -> classes 0..4 internally

Protocol:
    - One shared 80/20 development/test split stratified by original 5-class label
    - Final test set untouched during CV/model selection
    - 5-fold StratifiedKFold on development set
    - Standardization fitted on training fold only
    - Early stopping on fold validation loss
    - Unified metrics
    - Final model trained on development data with an internal validation split
    - Confusion matrices + training history + JSON/CSV results saved

Reference method:
    Adapted from public CNN-BiLSTM implementations for the UCI Epileptic
    Seizure Recognition dataset and the architecture idea in:
    "A Hybrid Deep Learning Approach for Epileptic Seizure Detection in EEG Signals"
    (IEEE JBHI, DOI: 10.1109/JBHI.2023.3265983)

This is an adaptation to the BS6200 common experimental protocol, not an
exact reproduction of the paper.
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# -----------------------------
# 1. Reproducibility
# -----------------------------
RANDOM_STATE = 6200

def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Better reproducibility. Can make GPU training slightly slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# 2. Dataset
# -----------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        # X shape: (N, 178)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 178)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# 3. CNN + BiLSTM model
# -----------------------------
class CNNBiLSTM(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.features = nn.Sequential(
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
            nn.MaxPool1d(kernel_size=2),
        )

        # After CNN: (batch, channels=128, reduced_time)
        # BiLSTM expects (batch, seq_len, input_size)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.bilstm(x)

        # h_n shape for bidirectional 1-layer LSTM: (2, batch, hidden)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat([h_forward, h_backward], dim=1)

        return self.classifier(h)


# -----------------------------
# 4. Utility functions
# -----------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    feature_cols = [f"x{i}" for i in range(1, 179)]
    missing_cols = [c for c in feature_cols + ["y"] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected columns: {missing_cols}\n"
            f"Found columns: {df.columns.tolist()[:10]} ... {df.columns.tolist()[-10:]}"
        )

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y5_original = df["y"].to_numpy(dtype=int)

    if not set(np.unique(y5_original)).issubset({1, 2, 3, 4, 5}):
        raise ValueError(f"Unexpected y labels: {np.unique(y5_original)}")

    y_binary = (y5_original == 1).astype(int)
    y_five = y5_original - 1  # 0..4 for CrossEntropyLoss

    return df, X, y_binary, y_five, y5_original


def make_class_weights(y, n_classes, device):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = len(y) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loaders(X_train, y_train, X_val, y_val, batch_size):
    train_ds = EEGDataset(X_train, y_train)
    val_ds = EEGDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


def predict(model, loader, device, n_classes):
    model.eval()
    all_y, all_prob = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.softmax(logits, dim=1)

            all_y.append(yb.numpy())
            all_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y)
    probs = np.concatenate(all_prob)
    y_pred = probs.argmax(axis=1)

    return y_true, y_pred, probs


def compute_metrics(y_true, y_pred, probs, task):
    if task == "binary":
        # class 1 = seizure
        pos_prob = probs[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision_seizure": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "recall_seizure": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "specificity": specificity,
            "f1_seizure": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "roc_auc": roc_auc_score(y_true, pos_prob),
            "pr_auc": average_precision_score(y_true, pos_prob),
        }

    # Multiclass
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        y_onehot = np.eye(5)[y_true]
        metrics["macro_roc_auc_ovr"] = roc_auc_score(
            y_onehot, probs, multi_class="ovr", average="macro"
        )
    except Exception:
        metrics["macro_roc_auc_ovr"] = np.nan

    return metrics


def plot_confusion_matrix(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def train_one_model(
    X_train,
    y_train,
    X_val,
    y_val,
    n_classes,
    device,
    epochs,
    batch_size,
    learning_rate,
    patience,
    use_class_weights=True,
):
    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val, batch_size
    )

    model = CNNBiLSTM(n_classes=n_classes).to(device)

    if use_class_weights:
        weights = make_class_weights(y_train, n_classes, device)
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    wait = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"  Epoch {epoch:03d} | "
                f"train loss={train_loss:.4f} | val loss={val_loss:.4f}"
            )

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)

    return model, history, best_epoch


def run_task(
    task,
    X,
    y,
    y5_original,
    dev_idx,
    test_idx,
    output_root,
    device,
    n_splits,
    epochs,
    batch_size,
    learning_rate,
    patience,
    use_class_weights,
):
    print("\n" + "=" * 80)
    print(f"RUNNING TASK: {task.upper()}")
    print("=" * 80)

    task_dir = output_root / task
    task_dir.mkdir(parents=True, exist_ok=True)

    n_classes = 2 if task == "binary" else 5

    X_dev = X[dev_idx]
    y_dev = y[dev_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Same development rows, but CV stratified by the task target.
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_metrics = []
    best_epochs = []

    for fold, (tr_local, va_local) in enumerate(skf.split(X_dev, y_dev), start=1):
        print(f"\n--- {task} | Fold {fold}/{n_splits} ---")

        X_tr_raw = X_dev[tr_local]
        y_tr = y_dev[tr_local]
        X_va_raw = X_dev[va_local]
        y_va = y_dev[va_local]

        # Leakage-free scaling: fit only on training fold.
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)

        set_seed(RANDOM_STATE + fold)

        model, history, best_epoch = train_one_model(
            X_tr,
            y_tr,
            X_va,
            y_va,
            n_classes=n_classes,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            use_class_weights=use_class_weights,
        )

        val_loader = DataLoader(
            EEGDataset(X_va, y_va),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        y_true, y_pred, probs = predict(
            model, val_loader, device, n_classes
        )

        metrics = compute_metrics(y_true, y_pred, probs, task)
        metrics["fold"] = fold
        metrics["best_epoch"] = best_epoch
        fold_metrics.append(metrics)
        best_epochs.append(best_epoch)

        print("  Fold metrics:")
        for k, v in metrics.items():
            if isinstance(v, (float, np.floating)):
                print(f"    {k}: {v:.4f}")

    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(task_dir / "cv_fold_metrics.csv", index=False)

    metric_cols = [
        c for c in fold_df.columns
        if c not in ["fold", "best_epoch"]
    ]
    summary = {}
    for c in metric_cols:
        summary[c] = {
            "mean": float(fold_df[c].mean()),
            "std": float(fold_df[c].std(ddof=1)),
        }

    print("\nCV summary (mean ± SD):")
    for k, v in summary.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # -------------------------
    # Final model
    # -------------------------
    # Internal validation split from DEVELOPMENT only.
    # Stratify using the task labels.
    dev_train_local, dev_val_local = train_test_split(
        np.arange(len(dev_idx)),
        test_size=0.15,
        stratify=y_dev,
        random_state=RANDOM_STATE
    )

    X_final_train_raw = X_dev[dev_train_local]
    y_final_train = y_dev[dev_train_local]
    X_final_val_raw = X_dev[dev_val_local]
    y_final_val = y_dev[dev_val_local]

    final_scaler = StandardScaler()
    X_final_train = final_scaler.fit_transform(X_final_train_raw)
    X_final_val = final_scaler.transform(X_final_val_raw)
    X_test_scaled = final_scaler.transform(X_test)

    set_seed(RANDOM_STATE + 999)

    print("\n--- Training final model using development data only ---")
    final_model, history, final_best_epoch = train_one_model(
        X_final_train,
        y_final_train,
        X_final_val,
        y_final_val,
        n_classes=n_classes,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        use_class_weights=use_class_weights,
    )

    test_loader = DataLoader(
        EEGDataset(X_test_scaled, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    y_true, y_pred, probs = predict(
        final_model, test_loader, device, n_classes
    )

    test_metrics = compute_metrics(y_true, y_pred, probs, task)

    print("\nFINAL TEST RESULTS:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save model
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "n_classes": n_classes,
            "task": task,
            "feature_names": [f"x{i}" for i in range(1, 179)],
            "scaler_mean": final_scaler.mean_,
            "scaler_scale": final_scaler.scale_,
        },
        task_dir / "final_model.pt"
    )

    # Save predictions
    pred_df = pd.DataFrame({
        "original_row_index": test_idx,
        "y_true_internal": y_true,
        "y_pred_internal": y_pred,
    })
    for c in range(n_classes):
        pred_df[f"prob_class_{c}"] = probs[:, c]
    pred_df.to_csv(task_dir / "final_test_predictions.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    if task == "binary":
        class_names = ["Non-seizure", "Seizure"]
    else:
        class_names = [
            "1 Seizure",
            "2 Tumor area",
            "3 Healthy brain area",
            "4 Eyes closed",
            "5 Eyes open",
        ]

    plot_confusion_matrix(
        cm,
        class_names,
        f"CNN-BiLSTM {task} - Final Test Confusion Matrix",
        task_dir / "confusion_matrix.png",
    )

    # Training curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_loss"], label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{task.capitalize()} final training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(task_dir / "training_history.png", dpi=200)
    plt.close(fig)

    result_payload = {
        "task": task,
        "cv_summary": summary,
        "final_test_metrics": {
            k: float(v) for k, v in test_metrics.items()
        },
        "final_best_epoch": int(final_best_epoch),
        "settings": {
            "random_state": RANDOM_STATE,
            "n_splits": n_splits,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "use_class_weights": use_class_weights,
        },
    }

    with open(task_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    return result_payload


def main():
    parser = argparse.ArgumentParser(
        description="BS6200 CNN-BiLSTM pipeline for UCI Epileptic Seizure Recognition."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data(1).csv",
        help="Path to CSV file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_cnn_bilstm",
        help="Output directory."
    )
    parser.add_argument(
        "--task",
        choices=["binary", "five", "both"],
        default="both",
        help="Which task(s) to run."
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds (default 5)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum epochs per fold (default 30)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience."
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable training-fold class weights."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: 2 folds, max 3 epochs."
    )

    args = parser.parse_args()

    set_seed()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find {csv_path.resolve()}\n"
            "Put this script in the same folder as data(1).csv, or pass --data PATH."
        )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {csv_path.resolve()}")
    df, X, y_binary, y_five, y5_original = load_data(csv_path)

    print(f"Dataset shape: {df.shape}")
    print(f"EEG feature matrix: {X.shape}")
    print("Original five-class counts:")
    print(pd.Series(y5_original).value_counts().sort_index())
    print("Binary counts [0=non-seizure, 1=seizure]:")
    print(pd.Series(y_binary).value_counts().sort_index())
    print(f"Missing EEG values: {np.isnan(X).sum()}")

    # One common split based on ORIGINAL 5-class labels.
    # This automatically preserves the binary seizure/non-seizure ratio as well.
    all_idx = np.arange(len(df))
    dev_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.20,
        stratify=y5_original,
        random_state=RANDOM_STATE,
    )

    split_df = pd.DataFrame({
        "row_index": np.concatenate([dev_idx, test_idx]),
        "split": (
            ["development"] * len(dev_idx)
            + ["final_test"] * len(test_idx)
        ),
    })
    split_df.to_csv(output_root / "shared_train_test_split.csv", index=False)

    print(
        f"\nShared split: development={len(dev_idx)}, "
        f"final_test={len(test_idx)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_splits = 2 if args.quick else args.folds
    epochs = 3 if args.quick else args.epochs

    if args.quick:
        print(
            "\nQUICK MODE ON: 2-fold CV and 3 epochs only. "
            "Use this only to verify that the code runs."
        )

    results = {}

    if args.task in ("binary", "both"):
        results["binary"] = run_task(
            task="binary",
            X=X,
            y=y_binary,
            y5_original=y5_original,
            dev_idx=dev_idx,
            test_idx=test_idx,
            output_root=output_root,
            device=device,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            use_class_weights=not args.no_class_weights,
        )

    if args.task in ("five", "both"):
        results["five"] = run_task(
            task="five",
            X=X,
            y=y_five,
            y5_original=y5_original,
            dev_idx=dev_idx,
            test_idx=test_idx,
            output_root=output_root,
            device=device,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            use_class_weights=not args.no_class_weights,
        )

    with open(
        output_root / "all_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"All outputs saved to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
