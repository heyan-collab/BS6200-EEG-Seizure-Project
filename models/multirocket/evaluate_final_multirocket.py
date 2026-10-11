"""One-time final-test evaluation for the frozen BS6200 MultiRocket model.

The final configuration is intentionally fixed in this script.  Probability
calibration is learned only from development-set out-of-fold decision scores;
the held-out final test is never used to fit or tune the calibrator.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

# Use a dedicated writable cache so an interrupted exploratory run cannot leave
# a lock that blocks the one-time final evaluation.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "bs6200_multirocket_final_cache"
_MPL_CACHE = _CACHE_ROOT / "matplotlib"
_NUMBA_CACHE = _CACHE_ROOT / "numba"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)
os.environ["NUMBA_CACHE_DIR"] = str(_NUMBA_CACHE)

import aeon
import numpy as np
import pandas as pd
import sklearn
from aeon.classification.convolution_based import MultiRocketClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from train_multirocket import (
    FEATURE_COLUMNS,
    binary_metrics,
    decision_scores,
    load_data,
    make_labels,
    multiclass_metrics,
    save_confusion_matrix,
    stratified_limit,
)


SEED = 6200
FINAL_N_KERNELS = 10_000
MAX_DILATIONS_PER_KERNEL = 32
N_FEATURES_PER_KERNEL = 4


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
        help="Shared split with fixed development/final_test membership",
    )
    parser.add_argument("--task", choices=["binary", "five_class"], required=True)
    parser.add_argument("--oof", default=None, help="Selected model's development OOF CSV")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use development fold 0 as a small pseudo-test; final_test is not used",
    )
    return parser.parse_args()


def default_oof_path(task: str) -> Path:
    run = "binary_cv5_k10000_final" if task == "binary" else "five_class_cv5_k10000_final"
    return Path(__file__).resolve().parent / "outputs" / run / "oof_predictions.csv"


def default_output_dir(task: str, smoke_test: bool) -> Path:
    name = f"{task}_{'final_smoke' if smoke_test else 'final_test'}"
    return Path(__file__).resolve().parent / "outputs" / name


def score_columns(task: str, class_labels: np.ndarray) -> list[str]:
    if task == "binary":
        return ["y_score"]
    return [f"score_class_{label}" for label in class_labels]


def fit_oof_calibrator(
    oof_path: Path,
    calibration_frame: pd.DataFrame,
    task: str,
    class_labels: np.ndarray,
) -> LogisticRegression:
    oof = pd.read_csv(oof_path)
    columns = score_columns(task, class_labels)
    required = {"row_index", "y_true", *columns}
    if not required <= set(oof.columns):
        raise ValueError(f"OOF file is missing columns: {sorted(required - set(oof.columns))}")
    if not oof["row_index"].is_unique:
        raise ValueError("OOF row_index is not unique")

    selected = calibration_frame[["row_index"]].merge(
        oof[["row_index", "y_true", *columns]],
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    if selected[["y_true", *columns]].isna().any().any():
        raise ValueError("OOF file does not cover every calibration sample")
    expected = make_labels(calibration_frame, task)
    if not np.array_equal(selected["y_true"].to_numpy(dtype=int), expected):
        raise ValueError("OOF labels do not match the shared development data")

    calibrator = LogisticRegression(max_iter=2_000, random_state=SEED)
    calibrator.fit(selected[columns].to_numpy(dtype=float), expected)
    return calibrator


def main() -> int:
    args = parse_args()
    print(f"Starting {args.task} {'smoke test' if args.smoke_test else 'final test'}...", flush=True)
    data_path = Path(args.data).resolve()
    split_path = Path(args.split).resolve()
    oof_path = Path(args.oof).resolve() if args.oof else default_oof_path(args.task)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else default_output_dir(args.task, args.smoke_test)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(data_path, split_path)
    print("Validated data and shared split.", flush=True)
    development = frame.loc[frame["split"].eq("development")].copy()
    final_test = frame.loc[frame["split"].eq("final_test")].copy()
    if len(development) != 9_200 or len(final_test) != 2_300:
        raise ValueError(
            f"Expected 9,200 development and 2,300 final-test rows; got "
            f"{len(development)} and {len(final_test)}"
        )

    if args.smoke_test:
        train_frame = development.loc[development["cv_fold"].ne(0)].copy()
        evaluation_frame = development.loc[development["cv_fold"].eq(0)].copy()
        y_train_all = make_labels(train_frame, args.task)
        y_evaluation_all = make_labels(evaluation_frame, args.task)
        train_frame, y_train = stratified_limit(train_frame, y_train_all, 400, SEED)
        evaluation_frame, y_evaluation = stratified_limit(
            evaluation_frame, y_evaluation_all, 200, SEED + 1
        )
        n_kernels = 100
        evaluation_name = "development_fold_0_smoke"
        final_test_accessed = False
    else:
        train_frame = development
        evaluation_frame = final_test
        y_train = make_labels(train_frame, args.task)
        y_evaluation = make_labels(evaluation_frame, args.task)
        n_kernels = FINAL_N_KERNELS
        evaluation_name = "final_test"
        final_test_accessed = True

    X_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[:, np.newaxis, :]
    X_evaluation = evaluation_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[:, np.newaxis, :]

    model = MultiRocketClassifier(
        n_kernels=n_kernels,
        max_dilations_per_kernel=MAX_DILATIONS_PER_KERNEL,
        n_features_per_kernel=N_FEATURES_PER_KERNEL,
        class_weight=None,
        n_jobs=args.n_jobs,
        random_state=SEED,
    )
    print(f"Fitting frozen MultiRocket configuration ({n_kernels} kernels)...", flush=True)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    print("Generating base predictions and decision scores...", flush=True)
    started = time.perf_counter()
    y_pred = model.predict(X_evaluation)
    score = decision_scores(model, X_evaluation)
    predict_seconds = time.perf_counter() - started
    class_labels = np.asarray(model.classes_)

    print("Fitting the development-OOF probability calibrator...", flush=True)
    calibrator = fit_oof_calibrator(
        oof_path,
        train_frame,
        args.task,
        class_labels,
    )
    score_matrix = score.reshape(-1, 1) if args.task == "binary" else score
    probabilities = calibrator.predict_proba(score_matrix)
    calibrated_prediction = calibrator.classes_[np.argmax(probabilities, axis=1)]
    print("Computing metrics and writing outputs...", flush=True)

    if args.task == "binary":
        metrics = binary_metrics(y_evaluation, y_pred, score)
        tn, fp, fn, tp = confusion_matrix(y_evaluation, y_pred, labels=[0, 1]).ravel()
        metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
        raw_score_frame = {"decision_score": score}
        labels = [0, 1]
    else:
        metrics = multiclass_metrics(
            y_evaluation,
            y_pred,
            score,
            class_labels,
        )
        raw_score_frame = {
            f"decision_score_class_{label}": score[:, position]
            for position, label in enumerate(class_labels)
        }
        labels = [1, 2, 3, 4, 5]

    metrics.update(
        {
            "task": args.task,
            "evaluation_set": evaluation_name,
            "n_train": len(train_frame),
            "n_evaluation": len(evaluation_frame),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "n_kernels": n_kernels,
            "class_weight": "none",
            "seed": SEED,
        }
    )
    pd.DataFrame([metrics]).to_csv(output_dir / "final_metrics.csv", index=False)

    predictions = pd.DataFrame(
        {
            "sample_id": evaluation_frame["sample_id"].to_numpy(),
            "row_index": evaluation_frame["row_index"].to_numpy(),
            "y_true": y_evaluation,
            "y_pred": y_pred,
            **raw_score_frame,
        }
    )
    predictions.to_csv(output_dir / "final_predictions.csv", index=False)

    probability_frame = pd.DataFrame(
        {
            "sample_id": evaluation_frame["sample_id"].to_numpy(),
            "row_index": evaluation_frame["row_index"].to_numpy(),
            "y_true": y_evaluation,
            "y_pred_base": y_pred,
            "y_pred_calibrated": calibrated_prediction,
            **{
                f"prob_class_{label}": probabilities[:, position]
                for position, label in enumerate(calibrator.classes_)
            },
        }
    )
    probability_frame.to_csv(output_dir / "final_probabilities.csv", index=False)

    matrix = confusion_matrix(y_evaluation, y_pred, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv",
        index_label="true_label",
    )
    save_confusion_matrix(
        y_evaluation,
        y_pred,
        labels,
        f"MultiRocket {args.task}, {evaluation_name}",
        output_dir / "confusion_matrix.png",
    )

    config = {
        **vars(args),
        "data": str(data_path),
        "split": str(split_path),
        "oof": str(oof_path),
        "output_dir": str(output_dir),
        "n_kernels_used": n_kernels,
        "max_dilations_per_kernel": MAX_DILATIONS_PER_KERNEL,
        "n_features_per_kernel": N_FEATURES_PER_KERNEL,
        "class_weight": "none",
        "seed": SEED,
        "calibration": "LogisticRegression fitted on development OOF decision scores",
        "metrics_use_base_predictions": True,
        "final_test_accessed": final_test_accessed,
        "python": platform.python_version(),
        "aeon": aeon.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    primary = metrics["f1"] if args.task == "binary" else metrics["macro_f1"]
    print(
        f"{evaluation_name}: task={args.task}, primary_F1={primary:.4f}, "
        f"fit={fit_seconds:.1f}s, evaluation={len(evaluation_frame)}"
    )
    print(f"Saved results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
