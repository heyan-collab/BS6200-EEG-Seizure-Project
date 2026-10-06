import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from train_cnn_transformer import (
    ROOT,
    SPLIT_PATH,
    SEED,
    DEVICE,
    LEARNING_RATE,
    CNNTransformer,
    set_seed,
    read_csv,
    load_data,
    make_loader,
    predict,
    calculate_metrics,
    save_plots,
)

OUTPUT_DIR = ROOT / "outputs" / "cnn_transformer_final"
FINAL_EPOCHS = {
    "binary": 22,
    "five_class": 32,
}


def run_final(task, X, original_y, train_idx, test_idx):
    set_seed(SEED)
    folder = OUTPUT_DIR / task
    folder.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally overwriting an existing final evaluation.
    if (folder / "metrics.json").exists():
        raise FileExistsError(
            f"Final results already exist in {folder}. "
            "Keep the existing evaluation."
        )

    epochs = FINAL_EPOCHS[task]
    num_classes = 2 if task == "binary" else 5
    y = (
        (original_y == 1).astype(np.int64)
        if task == "binary"
        else original_y.astype(np.int64) - 1
    )

    # Fit preprocessing on the full development set only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    train_loader = make_loader(X_train, y[train_idx], shuffle=True)

    model = CNNTransformer(num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    start = time.perf_counter()

    print(
        f"\nTask={task} | Device={DEVICE} | "
        f"Development samples={len(train_idx)} | "
        f"Fixed epochs={epochs}"
    )

    # Fixed training duration: no test-based early stopping.
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_idx)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
        })
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Training loss={train_loss:.4f}"
        )

    training_seconds = time.perf_counter() - start

    # Save the final model before evaluating the test set.
    torch.save({
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "num_classes": num_classes,
        "task": task,
        "epochs": epochs,
        "scaler_mean": torch.tensor(scaler.mean_),
        "scaler_scale": torch.tensor(scaler.scale_),
    }, folder / "final_model.pt")

    history = pd.DataFrame(history)
    history.to_csv(folder / "learning_curve.csv", index=False)

    # Evaluate the reserved test set after training is complete.
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    test_loader = make_loader(X_test, y[test_idx], shuffle=False)

    probabilities = predict(model, test_loader)
    metrics, predictions = calculate_metrics(
        y[test_idx], probabilities, task
    )

    # Reuse plotting utility; final training has no validation curve.
    plot_history = history.assign(val_primary=np.nan)
    save_plots(
        plot_history, y[test_idx], predictions, task, folder
    )

    # Replace the CV-style plot with a training-only curve.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_loss"])
    ax.set(
        xlabel="Epoch",
        ylabel="Training loss",
        title=f"{task}: full development training",
    )
    fig.tight_layout()
    fig.savefig(folder / "learning_curve.png", dpi=180)
    plt.close(fig)

    offset = 0 if task == "binary" else 1
    prediction_table = pd.DataFrame({
        "row_index": test_idx,
        "y_true": y[test_idx] + offset,
        "y_pred": predictions + offset,
    })
    for class_id in range(num_classes):
        prediction_table[f"prob_{class_id + offset}"] = (
            probabilities[:, class_id]
        )

    prediction_table.to_csv(
        folder / "test_predictions.csv", index=False
    )
    prediction_table.loc[
        prediction_table["y_true"] != prediction_table["y_pred"]
    ].to_csv(folder / "test_errors.csv", index=False)

    result = {
        "task": task,
        "epochs": epochs,
        "seed": SEED,
        "development_samples": len(train_idx),
        "test_samples": len(test_idx),
        "training_seconds": training_seconds,
        **metrics,
    }
    with open(folder / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"\n{task.upper()} FINAL TEST RESULTS")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return result


def main():
    # Reuse the original split validation.
    X, original_y, development = load_data()
    split = read_csv(SPLIT_PATH)
    split["row_index"] = pd.to_numeric(
        split["row_index"], errors="raise"
    ).astype(int)

    train_idx = development["row_index"].to_numpy()
    test_idx = split.loc[
        split["split"] == "final_test", "row_index"
    ].to_numpy()

    if len(train_idx) != 9200 or len(test_idx) != 2300:
        raise ValueError("Expected 9,200 development and 2,300 test samples.")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Development and test sets overlap.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for task in ["binary", "five_class"]:
        results.append(
            run_final(task, X, original_y, train_idx, test_idx)
        )

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "final_test_summary.csv", index=False
    )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()