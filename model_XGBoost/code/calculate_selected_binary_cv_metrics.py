"""Reproduce fixed binary CV and add metrics without tuning or test evaluation."""
import hashlib
import json
import random
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score)

from xgboost_baseline import FEATURES, read_csv


def main():
    root = Path(__file__).resolve().parent.parent
    out = root / 'results/binary/tuning'
    fold_path = out / 'selected_model_full_fold_metrics.csv'
    summary_path = out / 'selected_model_full_cv_summary.json'
    if fold_path.exists() or summary_path.exists():
        raise FileExistsError('New CV outputs already exist; refusing to overwrite')
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines()
    original_hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in tracked}
    chosen = json.loads((out/'best_parameters.json').read_text())
    old_summary = json.loads((out/'cv_summary.json').read_text())
    manifest = json.loads((out/'search_manifest.json').read_text())
    expected = dict(max_depth=6, learning_rate=0.2, n_estimators=300,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_alpha=0, reg_lambda=10)
    if chosen['parameters'] != expected or chosen['fixed_parameters']['random_state'] != 42:
        raise ValueError('Saved configuration differs from requested fixed configuration')
    if manifest['std_definition'] != 'population SD across five folds, ddof=0':
        raise ValueError('Unexpected existing SD convention')
    params = {**chosen['fixed_parameters'], **chosen['parameters']}
    random.seed(42); np.random.seed(42)
    split = read_csv(root/'data/shared_train_test_split.csv').set_index('row_index').sort_index()
    cv = read_csv(root/'data/shared_split_with_cv.csv').set_index('row_index').sort_index()
    if list(split.index) != list(range(11500)) or not split.split.equals(cv.split):
        raise ValueError('Invalid or inconsistent shared assignments')
    idx = split.index[split.split == 'development'].to_numpy(dtype=int)
    if len(idx) != 9200:
        raise ValueError('Expected exactly 9,200 development indices')
    # Exclude final-test rows while loading data, before any feature/label extraction.
    membership = set(idx)
    data = pd.read_csv(root/'data/data.csv', encoding='utf-8-sig',
        skiprows=lambda row: row > 0 and row-1 not in membership)
    data.index = idx
    X = data[FEATURES]
    y = (data.y.to_numpy() == 1).astype(int)
    folds = cv.loc[idx, 'cv_fold'].to_numpy()
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(folds).all() or set(folds) != set(range(5)):
        raise ValueError('Unexpected features or shared folds')
    if (y == 1).sum() != 1840:
        raise ValueError('Incorrect positive-class mapping or development membership')
    previous_assignments = pd.read_csv(out/'development_cv_assignments.csv')
    if not np.array_equal(idx, previous_assignments.row_index) or not np.array_equal(folds, previous_assignments.cv_fold):
        raise ValueError('Fold assignments differ from completed tuning run')
    rows = []
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        if len(train) != 7360 or len(valid) != 1840 or y[valid].sum() != 368:
            raise ValueError('Unexpected shared fold size/class balance')
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[train], y[train])
        if not np.array_equal(model.classes_, [0, 1]):
            raise ValueError('Unexpected binary probability column mapping')
        prob = model.predict_proba(X.iloc[valid])[:, 1]  # Validation only.
        pred = (prob >= chosen['threshold']).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[valid], pred, labels=[0, 1]).ravel()
        rows.append(dict(fold=fold,
            accuracy=float(accuracy_score(y[valid], pred)),
            balanced_accuracy=float(balanced_accuracy_score(y[valid], pred)),
            seizure_precision=float(precision_score(y[valid], pred, pos_label=1, zero_division=0)),
            seizure_recall=float(recall_score(y[valid], pred, pos_label=1, zero_division=0)),
            specificity=float(tn/(tn+fp)),
            seizure_f1=float(f1_score(y[valid], pred, pos_label=1, zero_division=0)),
            roc_auc=float(roc_auc_score(y[valid], prob)),
            pr_auc=float(average_precision_score(y[valid], prob, pos_label=1))))
        print(f'Completed fixed configuration fold {fold}: {rows[-1]}', flush=True)
    table = pd.DataFrame(rows)
    stats = {k: {'mean': float(table[k].mean()), 'sd': float(table[k].std(ddof=0))}
             for k in table.columns if k != 'fold'}
    # Fail before saving anything if even one old fold score or summary differs.
    previous = pd.read_csv(out/'fold_scores.csv')
    previous = previous[previous.candidate_id == chosen['candidate_id']].sort_values('fold')
    if len(previous) != 5 or list(previous.fold) != list(range(5)):
        raise ValueError('Missing original fixed-candidate fold scores')
    for key in ['accuracy', 'balanced_accuracy', 'seizure_f1']:
        if not np.allclose(table[key], previous[key], rtol=0, atol=1e-14):
            raise RuntimeError(f'Reproducibility failure for fold-level {key}; outputs not saved')
        for new_name, old_name in [('mean', 'mean'), ('sd', 'std')]:
            if not np.isclose(stats[key][new_name], old_summary[f'{old_name}_{key}'], rtol=0, atol=1e-14):
                raise RuntimeError(f'Reproducibility failure for {key} {new_name}; outputs not saved')
    for name, original in original_hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != original:
            raise RuntimeError(f'Existing file changed: {name}; outputs not saved')
    result = dict(metrics=stats, metadata=dict(std_ddof=0,
        std_convention='Population SD across 5 validation folds', n_folds=5,
        development_samples=9200, validation_samples_per_fold=1840,
        feature_set='F1 Raw EEG (x1–x178)', positive_class='Seizure = 1 (original y=1)',
        negative_class='Non-seizure = 0 (original y=2,3,4,5)',
        parameters=params, probability_threshold=chosen['threshold'],
        pr_auc_convention='sklearn average_precision_score, positive label 1',
        group_split_generation_seed=6200, model_seed=42,
        tuning_performed=False, model_selection_performed=False, final_test_used=False,
        existing_cv_reproducibility_checks_passed=True,
        versions={'xgboost':xgb.__version__, 'sklearn':sklearn.__version__, 'numpy':np.__version__, 'pandas':pd.__version__},
        input_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ['data/data.csv','data/shared_train_test_split.csv','data/shared_split_with_cv.csv',
             'results/binary/tuning/best_parameters.json','results/binary/tuning/cv_summary.json']}))
    table.to_csv(fold_path,index=False)
    summary_path.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: existing fold scores and summaries reproduced; population SD ddof=0. Saved only two new numerical files.')
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':main()
