"""Train and evaluate MultiRocket on the fixed BS6200 EEG folds.

The script intentionally evaluates development folds only. The final test set is
not accessed here and should remain locked until the group freezes all models.
"""

from __future__ import annotations

# Keep Matplotlib/Numba caches in a writable per-user temp directory.  This is
# especially important on managed Windows machines where the default profile
# cache can be read-only and make the first aeon import appear to hang.
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "bs6200_multirocket_cache"
_MPL_CACHE = _CACHE_ROOT / "matplotlib"
_NUMBA_CACHE = _CACHE_ROOT / "numba"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE))

import argparse
import json
import platform
import time
from pathlib import Path

import aeon
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from aeon.classification.convolution_based import MultiRocketClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 6200
FEATURE_COLUMNS = [f"X{i}" for i in range(1, 179)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Original EEG CSV")
    parser.add_argument(
        "--split",
        default=str(
            Path(__file__).resolve().parents[2]
            / "splits"
            / "shared_split_with_cv.csv"
        ),
        help="Shared split with fixed cv_fold values",
    )
    parser.add_argument("--task", choices=["binary", "five_class"], default="binary")
    parser.add_argument(
        "--fold",
        default="0",
        help="One fold number (0-4) or 'all'",
    )
    parser.add_argument("--n-kernels", type=int, default=1000)
    parser.add_argument("--max-dilations-per-kernel", type=int, default=32)
    parser.add_argument("--n-features-per-kernel", type=int, default=4)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use small stratified subsets to verify the pipeline quickly",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; defaults under models/multirocket/outputs",
    )
    return parser.parse_args()


def load_data(data_path: Path, split_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    split = pd.read_csv(split_path)

    missing_features = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing_features:
        raise ValueError(f"Missing EEG columns: {missing_features[:5]}")
    if "y" not in data.columns:
        raise ValueError("Original label column 'y' is missing")
    required_split = {"row_index", "split", "cv_fold"}
    if not required_split <= set(split.columns):
        raise ValueError(f"Split file must contain {sorted(required_split)}")
    if len(data) != len(split):
        raise ValueError(f"Data has {len(data)} rows but split has {len(split)} rows")
    if not split["row_index"].is_unique:
        raise ValueError("Split row_index is not unique")
    if set(split["row_index"]) != set(range(len(data))):
        raise ValueError("Split row_index does not exactly cover the original data")

    indexed = data.copy()
    indexed.insert(0, "row_index", np.arange(len(indexed)))
    merged = indexed.merge(split, on="row_index", how="inner", validate="one_to_one")
    if len(merged) != len(data):
        raise ValueError("Data/split merge lost rows")
    if merged[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Raw EEG features contain missing values")
    if "Unnamed" in merged.columns:
        if merged["Unnamed"].isna().any() or not merged["Unnamed"].is_unique:
            raise ValueError("The 'Unnamed' sample identifier is blank or non-unique")
        merged = merged.rename(columns={"Unnamed": "sample_id"})
    else:
        merged["sample_id"] = merged["row_index"].map(lambda value: f"row_{value}")
    return merged


def make_labels(frame: pd.DataFrame, task: str) -> np.ndarray:
    original = frame["y"].to_numpy(dtype=int)
    if task == "binary":
        return (original == 1).astype(int)
    return original


def stratified_limit(
    frame: pd.DataFrame,
    labels: np.ndarray,
    maximum: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(frame) <= maximum:
        return frame, labels
    positions = np.arange(len(frame))
    selected, _ = train_test_split(
        positions,
        train_size=maximum,
        stratify=labels,
        random_state=seed,
    )
    selected = np.sort(selected)
    return frame.iloc[selected].copy(), labels[selected]


def decision_scores(model: MultiRocketClassifier, X: np.ndarray) -> np.ndarray:
    """Return continuous Ridge decision scores, not aeon's fallback one-hot output."""
    if not hasattr(model, "pipeline_") or not hasattr(model.pipeline_, "decision_function"):
        raise AttributeError("Fitted MultiRocket pipeline has no decision_function")
    return np.asarray(model.pipeline_.decision_function(X))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
        "average_precision": average_precision_score(y_true, score),
    }


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    class_labels: np.ndarray,
) -> dict:
    if score.ndim != 2 or score.shape[1] != len(class_labels):
        raise ValueError("Multiclass decision-score columns do not match model classes")
    # sklearn's multiclass roc_auc_score requires probabilities that sum to one.
    # RidgeClassifierCV provides decision scores instead, so compute the standard
    # one-vs-rest AUC for each class directly and take their unweighted mean.
    per_class_auc = [
        roc_auc_score((y_true == class_label).astype(int), score[:, position])
        for position, class_label in enumerate(class_labels)
    ]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_roc_auc_ovr": float(np.mean(per_class_auc)),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_binary_curves(y_true: np.ndarray, score: np.ndarray, output_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, score)
    precision, recall, _ = precision_recall_curve(y_true, score)

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.plot(fpr, tpr, label=f"AUROC = {roc_auc_score(y_true, score):.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="MultiRocket ROC curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.plot(recall, precision, label=f"AP = {average_precision_score(y_true, score):.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", title="MultiRocket precision-recall curve")
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    data_path = Path(args.data).resolve()
    split_path = Path(args.split).resolve()
    run_name = f"{'smoke' if args.smoke_test else 'cv'}_{args.task}"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(__file__).resolve().parent / "outputs" / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(data_path, split_path)
    development = frame.loc[frame["split"].eq("development")].copy()
    if development["cv_fold"].isna().any():
        raise ValueError("A development sample has no cv_fold")
    development["cv_fold"] = development["cv_fold"].astype(int)

    folds = list(range(5)) if args.fold == "all" else [int(args.fold)]
    if any(fold not in range(5) for fold in folds):
        raise ValueError("fold must be 0, 1, 2, 3, 4, or all")

    all_predictions = []
    fold_metrics = []
    class_weight = None if args.class_weight == "none" else "balanced"

    for fold in folds:
        train_frame = development.loc[development["cv_fold"].ne(fold)].copy()
        validation_frame = development.loc[development["cv_fold"].eq(fold)].copy()
        y_train = make_labels(train_frame, args.task)
        y_validation = make_labels(validation_frame, args.task)

        if args.smoke_test:
            train_frame, y_train = stratified_limit(train_frame, y_train, 400, args.seed + fold)
            validation_frame, y_validation = stratified_limit(
                validation_frame, y_validation, 200, args.seed + 100 + fold
            )

        X_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[:, np.newaxis, :]
        X_validation = validation_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[:, np.newaxis, :]

        model = MultiRocketClassifier(
            n_kernels=args.n_kernels,
            max_dilations_per_kernel=args.max_dilations_per_kernel,
            n_features_per_kernel=args.n_features_per_kernel,
            class_weight=class_weight,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        y_pred = model.predict(X_validation)
        score = decision_scores(model, X_validation)
        predict_seconds = time.perf_counter() - started

        if args.task == "binary":
            metrics = binary_metrics(y_validation, y_pred, score)
            score_frame = {"y_score": score}
            labels = [0, 1]
        else:
            metrics = multiclass_metrics(
                y_validation,
                y_pred,
                score,
                model.classes_,
            )
            score_frame = {
                f"score_class_{class_label}": score[:, position]
                for position, class_label in enumerate(model.classes_)
            }
            labels = [1, 2, 3, 4, 5]

        metrics.update(
            {
                "fold": fold,
                "n_train": len(train_frame),
                "n_validation": len(validation_frame),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "n_kernels": args.n_kernels,
                "class_weight": args.class_weight,
                "smoke_test": args.smoke_test,
            }
        )
        fold_metrics.append(metrics)

        predictions = pd.DataFrame(
            {
                "sample_id": validation_frame["sample_id"].to_numpy(),
                "row_index": validation_frame["row_index"].to_numpy(),
                "fold": fold,
                "y_true": y_validation,
                "y_pred": y_pred,
                **score_frame,
            }
        )
        all_predictions.append(predictions)
        save_confusion_matrix(
            y_validation,
            y_pred,
            labels,
            f"MultiRocket {args.task}, fold {fold}",
            output_dir / f"confusion_matrix_fold_{fold}.png",
        )
        print(
            f"Fold {fold}: F1={metrics.get('f1', metrics.get('macro_f1')):.4f}, "
            f"fit={fit_seconds:.1f}s, validation={len(validation_frame)}"
        )

    metrics_frame = pd.DataFrame(fold_metrics).sort_values("fold")
    predictions_frame = pd.concat(all_predictions, ignore_index=True).sort_values("row_index")
    metrics_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions_frame.to_csv(output_dir / "oof_predictions.csv", index=False)

    numeric_metrics = metrics_frame.select_dtypes(include=[np.number]).drop(
        columns=["fold", "n_train", "n_validation", "n_kernels"], errors="ignore"
    )
    summary = pd.DataFrame(
        {"mean": numeric_metrics.mean(), "std": numeric_metrics.std(ddof=1)}
    )
    summary.to_csv(output_dir / "cv_summary.csv")

    combined_labels = [0, 1] if args.task == "binary" else [1, 2, 3, 4, 5]
    save_confusion_matrix(
        predictions_frame["y_true"].to_numpy(),
        predictions_frame["y_pred"].to_numpy(),
        combined_labels,
        f"MultiRocket {args.task}, combined validation predictions",
        output_dir / "confusion_matrix_combined.png",
    )
    if args.task == "binary":
        save_binary_curves(
            predictions_frame["y_true"].to_numpy(),
            predictions_frame["y_score"].to_numpy(),
            output_dir,
        )

    config = {
        **vars(args),
        "data": str(data_path),
        "split": str(split_path),
        "output_dir": str(output_dir),
        "python": platform.python_version(),
        "aeon": aeon.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "final_test_accessed": False,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
