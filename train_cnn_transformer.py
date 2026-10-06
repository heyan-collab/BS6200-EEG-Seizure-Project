import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------- Configuration ----------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "archive" / "Epileptic Seizure Recognition.csv"
SPLIT_PATH = ROOT / "splits" / "shared_split_with_cv.csv"
OUTPUT_DIR = ROOT / "outputs" / "cnn_transformer"



SEED = 6200
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
DEVICE = torch.device("cpu")
torch.set_num_threads(4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_csv(path):
    with open(path, "rb") as file:
        header = file.read(2)
    encoding = (
        "utf-16" if header in (b"\xff\xfe", b"\xfe\xff")
        else "utf-8-sig"
    )
    return pd.read_csv(path, encoding=encoding)


def load_data():
    data = read_csv(DATA_PATH)
    split = read_csv(SPLIT_PATH)

    columns = [f"X{i}" for i in range(1, 179)]
    X = data[columns].to_numpy(dtype=np.float32)
    original_y = data["y"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("EEG features contain missing or infinite values.")
    if not np.isin(original_y, [1, 2, 3, 4, 5]).all():
        raise ValueError("Unexpected original labels.")

    indices = pd.to_numeric(split["row_index"], errors="raise")
    if indices.isna().any() or (indices % 1 != 0).any():
        raise ValueError("row_index must contain integers.")
    split["row_index"] = indices.astype(int)

    if (
        len(split) != len(data)
        or split["row_index"].duplicated().any()
        or set(split["row_index"]) != set(range(len(data)))
    ):
        raise ValueError("Shared split must cover every data row exactly once.")

    if not split["split"].isin(["development", "final_test"]).all():
        raise ValueError("Unexpected split names.")

    development = split.loc[
        split["split"] == "development"
    ].copy()
    development["cv_fold"] = pd.to_numeric(
        development["cv_fold"], errors="raise"
    )
    if not development["cv_fold"].isin(range(5)).all():
        raise ValueError("Development fold IDs must be 0 to 4.")
    if set(development["cv_fold"]) != set(range(5)):
        raise ValueError("All five development folds must be present.")

    return X, original_y, development


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)).unsqueeze(1),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


class CNNTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

        # CNN reduces the sequence length from 178 to 44.
        # Position embeddings preserve temporal order.
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 44, 64)
        )
        nn.init.normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )

        # Independently initialize each encoder layer.
        for layer in self.transformer.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.zeros_(layer.self_attn.in_proj_bias)

            for linear in [
                layer.self_attn.out_proj,
                layer.linear1,
                layer.linear2,
            ]:
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)

        self.output_norm = nn.LayerNorm(64)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)                    # (batch, 64, 44)
        x = x.transpose(1, 2)              # (batch, 44, 64)
        x = x + self.position_embedding
        x = self.transformer(x)
        x = self.output_norm(x)

        mean_features = x.mean(dim=1)
        max_features = x.max(dim=1).values
        pooled = torch.cat(
            [mean_features, max_features], dim=1
        )                                 # (batch, 128)

        return self.classifier(pooled)


def predict(model, loader):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs.to(DEVICE))
            probabilities.append(
                torch.softmax(logits, dim=1).cpu().numpy()
            )
    return np.concatenate(probabilities)


def calculate_metrics(y, probabilities, task):
    if task == "binary":
        predictions = (probabilities[:, 1] >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "balanced_accuracy": balanced_accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, zero_division=0),
            "recall": recall_score(y, predictions, zero_division=0),
            "specificity": tn / (tn + fp),
            "f1": f1_score(y, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y, probabilities[:, 1]),
            # PR-AUC is reported using the average precision convention.
            "pr_auc_ap": average_precision_score(y, probabilities[:, 1]),
        }
    else:
        predictions = probabilities.argmax(axis=1)
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "balanced_accuracy": balanced_accuracy_score(y, predictions),
            "macro_precision": precision_score(
                y, predictions, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                y, predictions, average="macro", zero_division=0
            ),
            "macro_f1": f1_score(
                y, predictions, average="macro", zero_division=0
            ),
            "macro_roc_auc_ovr": roc_auc_score(
                y, probabilities, multi_class="ovr", average="macro"
            ),
        }
    return metrics, predictions


def save_plots(history, y, predictions, task, folder):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"])
    axes[0].set(xlabel="Epoch", ylabel="Training loss")
    axes[1].plot(history["epoch"], history["val_primary"])
    axes[1].set(
        xlabel="Epoch",
        ylabel="Seizure F1" if task == "binary" else "Macro F1",
    )
    fig.tight_layout()
    fig.savefig(folder / "learning_curve.png", dpi=180)
    plt.close(fig)

    labels = [0, 1] if task == "binary" else list(range(5))
    names = ["Non-seizure", "Seizure"] if task == "binary" else [
        str(i) for i in range(1, 6)
    ]
    cm = confusion_matrix(y, predictions, labels=labels)
    pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in names],
        columns=[f"pred_{name}" for name in names],
    ).to_csv(folder / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"{task}: best checkpoint")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def run_fold(X, y, development, task, fold):
    set_seed(SEED + fold)
    folder = OUTPUT_DIR / task / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)

    train_idx = development.loc[
        development["cv_fold"] != fold, "row_index"
    ].to_numpy()
    val_idx = development.loc[
        development["cv_fold"] == fold, "row_index"
    ].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)

    train_loader = make_loader(X_train, y[train_idx], True)
    val_loader = make_loader(X_val, y[val_idx], False)

    num_classes = 2 if task == "binary" else 5
    model = CNNTransformer(num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    primary = "f1" if task == "binary" else "macro_f1"
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    start = time.perf_counter()

    print(f"\nTask={task} | Fold={fold} | Device={DEVICE}")

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)

        probabilities = predict(model, val_loader)
        metrics, _ = calculate_metrics(y[val_idx], probabilities, task)
        score = metrics[primary]
        train_loss = total_loss / len(train_idx)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_primary": score,
            **{f"val_{key}": value for key, value in metrics.items()},
        })

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:02d} | Loss={train_loss:.4f} | "
            f"Val {primary}={score:.4f} | "
            f"Time={time.perf_counter() - epoch_start:.1f}s"
        )
        if stale_epochs >= PATIENCE:
            print(f"Early stopping. Best epoch: {best_epoch}")
            break

    model.load_state_dict(best_state)
    probabilities = predict(model, val_loader)
    metrics, predictions = calculate_metrics(
        y[val_idx], probabilities, task
    )

    # Save the checkpoint together with its training-only scaler.
    torch.save({
        "model_state_dict": best_state,
        "num_classes": num_classes,
        "task": task,
        "fold": fold,
        "best_epoch": best_epoch,
        "scaler_mean": torch.tensor(scaler.mean_),
        "scaler_scale": torch.tensor(scaler.scale_),
    }, folder / "best_model.pt")

    history = pd.DataFrame(history)
    history.to_csv(folder / "learning_curve.csv", index=False)
    save_plots(history, y[val_idx], predictions, task, folder)

    # Five-class exports use the original labels 1 to 5.
    offset = 0 if task == "binary" else 1
    prediction_table = pd.DataFrame({
        "row_index": val_idx,
        "cv_fold": fold,
        "y_true": y[val_idx] + offset,
        "y_pred": predictions + offset,
    })
    for class_id in range(num_classes):
        prediction_table[f"prob_{class_id + offset}"] = probabilities[:, class_id]
    prediction_table.to_csv(folder / "predictions.csv", index=False)

    result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "training_seconds": time.perf_counter() - start,
        **metrics,
    }
    with open(folder / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Best validation {primary}: {best_score:.4f}")
    return result, prediction_table


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, original_y, development = load_data()

    config = {
        "model": "CNN-Transformer",
        "cnn_channels": [32, 64],
        "kernel_sizes": [7, 5],
        "batch_normalization": True,
        "d_model": 64,
        "attention_heads": 4,
        "transformer_layers": 2,
        "feedforward_dimension": 128,
        "position_encoding": "learnable",
        "temporal_pooling": "mean + max",
        "transformer_dropout": 0.1,
        "classifier_dropout": 0.2,
        "dropout": 0.2,
        "feature_set": "F1: X1-X178",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "scaling": "Per-feature StandardScaler fitted on each training fold",
        "binary_threshold": 0.5,
        "pr_auc_convention": "Average precision",
        "multiclass_auc_convention": "Macro one-vs-rest",
    }
    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    for task in ["binary"]:
        y = (
            (original_y == 1).astype(np.int64)
            if task == "binary"
            else original_y.astype(np.int64) - 1
        )
        results, predictions = [], []

        for fold in range(5):
            result, prediction_table = run_fold(
                X, y, development, task, fold
            )
            results.append(result)
            predictions.append(prediction_table)

        task_dir = OUTPUT_DIR / task
        results = pd.DataFrame(results)
        results.to_csv(task_dir / "cv_results.csv", index=False)

        metric_columns = [
            column for column in results.columns
            if column not in ["fold", "best_epoch", "training_seconds"]
        ]
        summary = results[metric_columns].agg(["mean", "std"]).T
        summary.index.name = "metric"
        summary.to_csv(task_dir / "cv_summary.csv")

        pd.concat(predictions, ignore_index=True).sort_values(
            "row_index"
        ).to_csv(task_dir / "oof_predictions.csv", index=False)

        print(f"\n{task.upper()} CV SUMMARY")
        print(summary.round(4).to_string())

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("Final test evaluation has not been performed.")


if __name__ == "__main__":
    main()