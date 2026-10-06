from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from train_cnn_transformer import ROOT, CNNTransformer, read_csv
    
FINAL_DIR = ROOT / "outputs" / "cnn_transformer_final"
WINDOW_SIZE = 12
STRIDE = 6


def get_probabilities(model, inputs):
    with torch.no_grad():
        tensor = torch.from_numpy(
            inputs.astype(np.float32)
        ).unsqueeze(1)
        return torch.softmax(model(tensor), dim=1).numpy()


def occlusion_importance(model, signal, target_class):
    """
    Replace overlapping windows with zero in standardized space.
    Zero corresponds to the development-set mean at each position.

    Positive importance means masking that window reduces the
    probability of the originally predicted class.
    """
    baseline = get_probabilities(
        model, signal[None, :]
    )[0, target_class]

    starts = list(range(0, len(signal) - WINDOW_SIZE + 1, STRIDE))
    last_start = len(signal) - WINDOW_SIZE
    if starts[-1] != last_start:
        starts.append(last_start)

    masked_signals = np.repeat(
        signal[None, :], len(starts), axis=0
    )
    for i, start in enumerate(starts):
        masked_signals[i, start:start + WINDOW_SIZE] = 0.0

    masked_probabilities = get_probabilities(
        model, masked_signals
    )[:, target_class]
    drops = baseline - masked_probabilities

    importance = np.zeros(len(signal), dtype=np.float64)
    counts = np.zeros(len(signal), dtype=np.float64)

    for start, drop in zip(starts, drops):
        importance[start:start + WINDOW_SIZE] += drop
        counts[start:start + WINDOW_SIZE] += 1

    return importance / np.maximum(counts, 1), baseline


def save_confusion_matrix(predictions, task, folder):
    labels = [0, 1] if task == "binary" else [1, 2, 3, 4, 5]
    names = (
        ["Non-seizure", "Seizure"]
        if task == "binary"
        else [str(label) for label in labels]
    )

    matrix = confusion_matrix(
        predictions["y_true"],
        predictions["y_pred"],
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        matrix, display_labels=names
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{task}: final test confusion matrix")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def select_examples(predictions, task):
    """
    Deterministic descriptive examples.
    Select the lowest row_index in each available category.
    """
    correct = predictions["y_true"] == predictions["y_pred"]

    if task == "binary":
        categories = [
            ("correct_seizure",
             correct & (predictions["y_true"] == 1)),
            ("correct_non_seizure",
             correct & (predictions["y_true"] == 0)),
            ("false_negative",
             (predictions["y_true"] == 1)
             & (predictions["y_pred"] == 0)),
            ("false_positive",
             (predictions["y_true"] == 0)
             & (predictions["y_pred"] == 1)),
        ]
    else:
        categories = [
            ("correct_class_1",
             correct & (predictions["y_true"] == 1)),
            ("correct_class_3",
             correct & (predictions["y_true"] == 3)),
            ("class_3_predicted_as_2",
             (predictions["y_true"] == 3)
             & (predictions["y_pred"] == 2)),
            ("class_4_predicted_as_5",
             (predictions["y_true"] == 4)
             & (predictions["y_pred"] == 5)),
        ]

    examples = []
    for name, mask in categories:
        candidates = predictions.loc[mask].sort_values("row_index")
        if not candidates.empty:
            examples.append((name, candidates.iloc[0]))
    return examples


def analyze_task(data, task):
    folder = FINAL_DIR / task
    analysis_dir = folder / "interpretation"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        folder / "final_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = CNNTransformer(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["scaler_mean"].numpy()
    scale = checkpoint["scaler_scale"].numpy()

    predictions = pd.read_csv(folder / "test_predictions.csv")
    save_confusion_matrix(predictions, task, folder)

    feature_columns = [f"X{i}" for i in range(1, 179)]
    all_signals = data[feature_columns].to_numpy(dtype=np.float32)
    examples = select_examples(predictions, task)
    summary = []

    for name, row in examples:
        row_index = int(row["row_index"])
        true_label = int(row["y_true"])
        predicted_label = int(row["y_pred"])

        raw_signal = all_signals[row_index]
        standardized = (
            (raw_signal - mean) / scale
        ).astype(np.float32)

        target_class = (
            predicted_label
            if task == "binary"
            else predicted_label - 1
        )

        # Explain the model's prediction, including incorrect predictions.
        importance, probability = occlusion_importance(
            model, standardized, target_class
        )

        # Verify that the loaded checkpoint reproduces the saved prediction.
        probabilities = get_probabilities(
            model, standardized[None, :]
        )[0]
        reproduced_label = (
            int(probabilities[1] >= 0.5)
            if task == "binary"
            else int(probabilities.argmax()) + 1
        )
        if reproduced_label != predicted_label:
            raise ValueError(
                f"Prediction mismatch for row {row_index}. "
                "Check the data version and model checkpoint."
            )

        x = np.arange(1, 179)
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True
        )

        axes[0].plot(x, raw_signal, color="steelblue")
        axes[0].set_ylabel("Raw EEG value")
        axes[0].set_title(
            f"{task} | {name} | row_index={row_index}\n"
            f"True={true_label}, Predicted={predicted_label}, "
            f"Predicted-class probability={probability:.3f}"
        )

        axes[1].plot(x, importance, color="darkorange")
        axes[1].axhline(0, color="gray", linestyle="--")
        axes[1].fill_between(
            x, importance, 0,
            where=importance > 0,
            color="darkorange", alpha=0.25,
        )
        axes[1].set_xlabel("Sample position (X1-X178)")
        axes[1].set_ylabel("Probability drop")
        axes[1].set_title(
            "Occlusion sensitivity: positive values support "
            "the original prediction"
        )

        fig.tight_layout()
        fig.savefig(analysis_dir / f"{name}.png", dpi=200)
        plt.close(fig)

        pd.DataFrame({
            "sample_position": x,
            "raw_eeg": raw_signal,
            "occlusion_importance": importance,
        }).to_csv(analysis_dir / f"{name}.csv", index=False)

        summary.append({
            "example": name,
            "row_index": row_index,
            "y_true": true_label,
            "y_pred": predicted_label,
            "predicted_class_probability": probability,
        })

    pd.DataFrame(summary).to_csv(
        analysis_dir / "example_summary.csv", index=False
    )
    print(f"{task}: saved {len(examples)} explanation plots.")
    print(f"Output directory: {analysis_dir}")


if __name__ == "__main__":
    data = read_csv(
        ROOT / "archive" / "Epileptic Seizure Recognition.csv"
    )
    for task in ["binary", "five_class"]:
        analyze_task(data, task)

    print("Analysis complete. No model retraining was performed.")