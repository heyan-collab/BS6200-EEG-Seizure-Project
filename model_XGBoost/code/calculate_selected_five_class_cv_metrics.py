"""Supplement fixed five-class shared CV metrics; never evaluate the final model."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score)

from xgboost_baseline import FEATURES, read_csv


def main():
    root = Path(__file__).resolve().parent.parent
    out = root/'results/tuning'
    outputs = [out/'selected_model_full_fold_metrics.csv', out/'selected_model_full_cv_summary.json']
    if any(p.exists() for p in outputs):
        raise FileExistsError('Supplementary CV outputs already exist; refusing to overwrite')
    # Protect tracked and untracked prior work, including the preceding binary supplement.
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file() and '.git' not in p.relative_to(root).parts
        and '__pycache__' not in p.relative_to(root).parts}
    best = json.loads((out/'best_parameters.json').read_text())
    expected = dict(max_depth=8, learning_rate=0.1, n_estimators=200,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_alpha=0, reg_lambda=1)
    if best['parameters'] != expected or best['fixed_parameters']['random_state'] != 42:
        raise ValueError('Saved settings differ from the requested fixed configuration')
    params = {**best['fixed_parameters'], **best['parameters']}
    random.seed(42); np.random.seed(42)
    split = read_csv(root/'data/shared_train_test_split.csv').set_index('row_index').sort_index()
    cv = read_csv(root/'data/shared_split_with_cv.csv').set_index('row_index').sort_index()
    if list(split.index) != list(range(11500)) or list(cv.index) != list(range(11500)) or not split.split.equals(cv.split):
        raise ValueError('Shared index coverage/split labels are inconsistent')
    idx = split.index[split.split == 'development'].to_numpy(dtype=int)
    membership = set(idx)
    if len(idx) != 9200:raise ValueError('Expected 9,200 development rows')
    # Skip all final-test records while loading; no final-test feature/label extraction.
    data = pd.read_csv(root/'data/data.csv', encoding='utf-8-sig',
        skiprows=lambda row: row > 0 and row-1 not in membership)
    data.index = idx
    X = data[FEATURES]; y = data.y.to_numpy(dtype=int)
    folds = cv.loc[idx,'cv_fold'].to_numpy()
    assignments = pd.read_csv(out/'development_cv_assignments.csv')
    if not np.array_equal(idx, assignments.row_index) or not np.array_equal(folds, assignments.cv_fold):
        raise ValueError('Shared assignments differ from completed five-class tuning')
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(folds).all() or set(folds) != set(range(5)) or set(y) != set(range(1,6)):
        raise ValueError('Invalid development features, labels or folds')
    rows = []
    for fold in range(5):
        train = np.flatnonzero(folds != fold); valid = np.flatnonzero(folds == fold)
        if len(train) != 7360 or len(valid) != 1840 or any((y[valid]==c).sum()!=368 for c in range(1,6)):
            raise ValueError('Unexpected fold size or class distribution')
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[train], y[train]-1)
        if model.get_booster().feature_names != FEATURES or not np.array_equal(model.classes_,range(5)):
            raise ValueError('Unexpected model feature/output mapping')
        # Match the original tuning prediction path for exact reproducibility.
        pred = model.predict(X.iloc[valid]).astype(int)+1
        prob = model.predict_proba(X.iloc[valid])
        if prob.shape != (1840,5) or not np.allclose(prob.sum(axis=1),1,atol=1e-6) or not np.array_equal(prob.argmax(axis=1)+1,pred):
            raise ValueError('Validation probability output mapping is inconsistent')
        precision, recall, f1, _ = precision_recall_fscore_support(y[valid],pred,
            labels=[1,2,3,4,5],average='macro',zero_division=0)
        rows.append(dict(fold=fold, accuracy=float(accuracy_score(y[valid],pred)),
            balanced_accuracy=float(balanced_accuracy_score(y[valid],pred)),
            macro_precision=float(precision), macro_recall=float(recall), macro_f1=float(f1),
            macro_roc_auc=float(roc_auc_score(y[valid],prob,labels=[1,2,3,4,5],average='macro',multi_class='ovr'))))
        print(f'Completed fixed configuration fold {fold}: {rows[-1]}',flush=True)
    table = pd.DataFrame(rows)
    stats = {k:dict(mean=float(table[k].mean()),sd=float(table[k].std(ddof=0))) for k in table.columns if k!='fold'}
    previous = pd.read_csv(out/'fold_scores.csv')
    previous = previous[previous.candidate_id==best['candidate_id']].sort_values('fold')
    candidate = pd.read_csv(out/'candidate_scores.csv')
    candidate = candidate[candidate.candidate_id==best['candidate_id']]
    if len(previous)!=5 or list(previous.fold)!=list(range(5)) or len(candidate)!=1:
        raise ValueError('Existing selected-candidate CV records incomplete')
    for metric in ['accuracy','macro_f1']:
        if not np.allclose(table[metric],previous[metric],rtol=0,atol=1e-14):
            raise RuntimeError(f'Reproducibility mismatch: fold {metric}; nothing saved')
        for current,old in [('mean','mean'),('sd','std')]:
            if not np.isclose(stats[metric][current],best[f'{old}_cv_{metric}'],rtol=0,atol=1e-14) or not np.isclose(stats[metric][current],candidate.iloc[0][f'{old}_{metric}'],rtol=0,atol=1e-14):
                raise RuntimeError(f'Reproducibility mismatch: {metric} {current}; nothing saved')
    # Read saved final scores only: no final model load, inference or re-evaluation.
    final_path = root/'results/final/five_class_protocol_metrics.json'
    final = json.loads(final_path.read_text())
    original = json.loads((root/'results/final/metrics.json').read_text())
    expected_final = dict(accuracy=0.701304,balanced_accuracy=0.701304,
        macro_precision=0.703851,macro_recall=0.701304,macro_f1=0.700904,macro_roc_auc=0.924832)
    for key,value in expected_final.items():
        if not np.isclose(final[key],value,atol=5e-7,rtol=0):raise RuntimeError('Saved final metric does not match expected value')
        if key in original['test'] and original['test'][key]!=final[key]:raise RuntimeError('Saved final files disagree')
    paths = dict(confusion_matrix='results/final/confusion_matrix_test.csv',
        predictions='results/final/predictions.csv',
        probabilities='results/final/five_class_protocol_test_probabilities.csv')
    if not all((root/p).exists() for p in paths.values()):raise FileNotFoundError('Missing saved final artifacts')
    for path,h in protected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=h:
            raise RuntimeError(f'Previously existing file changed: {path}; nothing saved')
    result = dict(metrics=stats,metadata=dict(std_ddof=0,std_convention='Population SD across five folds',
        development_samples=9200,n_folds=5,feature_set='F1 Raw EEG (x1–x178)',class_order=[1,2,3,4,5],
        label_encoding='Model outputs 0–4 map to original Classes 1–5',parameters=params,
        macro_roc_auc_convention='sklearn average=macro, multi_class=ovr, labels=[1,2,3,4,5]',
        group_split_generation_seed=6200,model_seed=42,tuning_performed=False,model_selection_performed=False,
        final_test_used_for_cv=False,final_model_retrained=False,existing_cv_reproducibility_checks_passed=True,
        saved_final_metrics_verified={k:final[k] for k in expected_final},saved_final_artifact_paths=paths,
        versions={'xgboost':xgb.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [root/'data/data.csv',root/'data/shared_train_test_split.csv',root/'data/shared_split_with_cv.csv',out/'best_parameters.json',final_path]}))
    table.to_csv(outputs[0],index=False);outputs[1].write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: existing fold scores and CV summaries reproduced; all prior files unchanged. Saved supplementary CV files only.',flush=True)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
