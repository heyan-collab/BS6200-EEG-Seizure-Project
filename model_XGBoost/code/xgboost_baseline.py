"""Fixed five-class EEG baseline; the supplied test set is evaluation-only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

SEED = 42
FEATURES = [f"x{i}" for i in range(1, 179)]
CLASSES = [1, 2, 3, 4, 5]


def read_csv(path):
    """Support the supplied UTF-16 split file and UTF-8 data file."""
    with path.open("rb") as handle:
        bom = handle.read(2)
    return pd.read_csv(path, encoding="utf-16" if bom in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig")


def evaluate(y_true, y_pred):
    """Report metrics with original labels 1–5 and zero-division handling."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
    )
    return {
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "baseline")
    args = parser.parse_args()

    # 1. Read data and validate the complete positional split before indexing.
    data = read_csv(args.data_dir / "data.csv")
    split = read_csv(args.data_dir / "shared_train_test_split.csv")
    if list(split.columns) != ["row_index", "split"]:
        raise ValueError("Unexpected split columns")
    index = pd.to_numeric(split["row_index"], errors="raise")
    if index.isna().any() or not np.isfinite(index).all() or (index % 1 != 0).any():
        raise ValueError("row_index must contain finite integers")
    index = index.astype(int)
    if len(index) != len(data) or index.duplicated().any() or set(index) != set(range(len(data))):
        raise ValueError("Split must cover every data row exactly once")
    if set(split["split"]) != {"development", "final_test"}:
        raise ValueError("Unexpected split labels")
    if not set(FEATURES + ["y"]).issubset(data.columns):
        raise ValueError("Missing EEG features or target")
    X = data[FEATURES].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(data["y"], errors="raise")
    if not np.isfinite(X.to_numpy()).all() or y.isna().any() or set(y) != set(CLASSES):
        raise ValueError("Invalid features or five-class labels")

    # 2. Honor row_index, not split-file order. No random split or CV is used.
    train_idx = index[split["split"] == "development"].to_numpy()
    test_idx = index[split["split"] == "final_test"].to_numpy()
    if set(y.iloc[train_idx]) != set(CLASSES):
        raise ValueError("Development set must contain all five classes")

    # 3. Fixed, untuned CPU baseline. XGBoost requires zero-based class labels.
    model = XGBClassifier(
        objective="multi:softprob", num_class=5, n_estimators=100,
        max_depth=6, learning_rate=0.3, subsample=1.0,
        colsample_bytree=1.0, tree_method="hist", device="cpu",
        eval_metric="mlogloss", random_state=SEED, n_jobs=2,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx].to_numpy(dtype=int) - 1)

    # 4. Evaluate only after fitting. Test data never enters fit or an eval_set.
    results = {}
    predictions = []
    for name, rows in [("training", train_idx), ("test", test_idx)]:
        pred = model.predict(X.iloc[rows]).astype(int) + 1
        results[name] = evaluate(y.iloc[rows], pred)
        predictions.append(pd.DataFrame({
            "row_index": rows, "split": name,
            "y_true": y.iloc[rows].to_numpy(), "y_pred": pred,
        }))
        print(f"\n{name.upper()} ({len(rows)} samples)")
        print(json.dumps({k: v for k, v in results[name].items() if k not in ("classification_report", "confusion_matrix")}, indent=2))
        print("Confusion matrix: rows=true, columns=predicted; labels=1,2,3,4,5")
        print(np.array(results[name]["confusion_matrix"]))
        print(classification_report(y.iloc[rows], pred, labels=CLASSES, digits=4, zero_division=0))

    # 5. Save metrics, predictions, model and input hashes for reproducibility.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results["metadata"] = {
        "seed": SEED, "features": FEATURES, "class_order": CLASSES,
        "parameters": model.get_params(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "input_sha256": {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() for name in ["data.csv", "shared_train_test_split.csv"]},
        "limitation": "Shared split is retained exactly. Identifier-suffix groups cross train/test; same-source leakage remains possible. No tuning or cross-validation performed.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    pd.concat(predictions).sort_values("row_index").to_csv(args.output_dir / "predictions.csv", index=False)
    # Native booster export avoids sklearn wrapper metadata compatibility issues.
    model.get_booster().save_model(args.output_dir / "model.json")


if __name__ == "__main__":
    main()
