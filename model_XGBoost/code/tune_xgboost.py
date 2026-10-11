"""Random-search XGBoost using only development rows and the provided CV folds."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

SEED = 42
FEATURES = [f'x{i}' for i in range(1, 179)]
SPACE = {
    'max_depth': [3, 4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5, 10],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'reg_alpha': [0.0, 0.1, 1.0],
    'reg_lambda': [1.0, 3.0, 10.0],
}


def read_csv(path):
    with path.open('rb') as f:
        bom = f.read(2)
    return pd.read_csv(path, encoding='utf-16' if bom in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig')


def load_development(data_dir):
    """Validate split metadata, then select development rows before extracting X/y."""
    split = read_csv(data_dir / 'shared_train_test_split.csv')
    cv = read_csv(data_dir / 'shared_split_with_cv.csv')
    data = read_csv(data_dir / 'data.csv')
    for frame in (split, cv):
        idx = pd.to_numeric(frame['row_index'], errors='raise')
        if idx.isna().any() or not np.isfinite(idx).all() or (idx % 1 != 0).any():
            raise ValueError('Invalid row indices')
        frame['row_index'] = idx.astype(int)
        if len(frame) != len(data) or idx.duplicated().any() or set(idx) != set(range(len(data))):
            raise ValueError('Split must cover each data row exactly once')
        if set(frame['split']) != {'development', 'final_test'}:
            raise ValueError('Unexpected split labels')
    split = split.set_index('row_index').sort_index()
    cv = cv.set_index('row_index').sort_index()
    if not split['split'].equals(cv['split']):
        raise ValueError('Shared train/test and CV split disagree')
    dev_idx = split.index[split['split'] == 'development'].to_numpy()
    test_idx = split.index[split['split'] == 'final_test'].to_numpy()
    if len(dev_idx) != 9200 or len(test_idx) != 2300:
        raise ValueError('Expected 9,200 development and 2,300 held-out rows')
    if cv.loc[test_idx, 'cv_fold'].notna().any():
        raise ValueError('Final test rows must not participate in CV')
    folds = pd.to_numeric(cv.loc[dev_idx, 'cv_fold'], errors='raise').to_numpy()
    if not np.isfinite(folds).all() or set(folds) != set(range(5)):
        raise ValueError('Expected the provided five folds 0–4')
    development = data.iloc[dev_idx]
    X = development[FEATURES].apply(pd.to_numeric, errors='raise')
    y = pd.to_numeric(development['y'], errors='raise').to_numpy()
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y).all() or set(y) != {1, 2, 3, 4, 5}:
        raise ValueError('Invalid development features or labels')
    for fold in range(5):
        if np.sum(folds == fold) != 1840 or set(y[folds == fold]) != {1, 2, 3, 4, 5}:
            raise ValueError('Unexpected validation fold size or classes')
    return X, y.astype(int) - 1, folds.astype(int), dev_idx


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--output-dir', type=Path, default=root / 'results/tuning')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use an empty output directory to preserve prior results')
    random.seed(SEED)
    np.random.seed(SEED)
    X, y, folds, dev_idx = load_development(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(ParameterSampler(SPACE, n_iter=12, random_state=SEED))
    fixed = dict(objective='multi:softprob', num_class=5, tree_method='hist',
                 device='cpu', eval_metric='mlogloss', random_state=SEED, n_jobs=4)
    manifest = {
        'seed': SEED, 'strategy': '12 configurations sampled without replacement from discrete ranges',
        'selection_metric': 'mean 5-fold macro F1; exact ties resolved by candidate order',
        'std_definition': 'population standard deviation across five folds (ddof=0)',
        'search_space': SPACE, 'candidates': candidates, 'fixed_parameters': fixed,
        'features': FEATURES, 'target': 'y (1–5, encoded internally as 0–4)',
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'input_sha256': {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() for name in ['data.csv', 'shared_train_test_split.csv', 'shared_split_with_cv.csv']},
        'final_test_evaluated': False,
        'limitation': 'Shared folds overlap in inferred recording groups; CV estimates do not demonstrate generalization to unseen recordings or subjects.',
    }
    (args.output_dir / 'search_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    pd.DataFrame({'row_index': dev_idx, 'cv_fold': folds}).to_csv(args.output_dir / 'development_cv_assignments.csv', index=False)
    fold_rows, summary = [], []
    for candidate_id, params in enumerate(candidates, 1):
        scores = []
        for fold in range(5):
            train = np.flatnonzero(folds != fold)
            valid = np.flatnonzero(folds == fold)
            # Fresh model per fold; no test eval_set, early stopping or generated folds.
            model = XGBClassifier(**fixed, **params)
            model.fit(X.iloc[train], y[train])
            pred = model.predict(X.iloc[valid])
            macro_f1 = float(f1_score(y[valid], pred, labels=range(5), average='macro', zero_division=0))
            accuracy = float(accuracy_score(y[valid], pred))
            row = dict(candidate_id=candidate_id, fold=fold, n_train=len(train), n_validation=len(valid), macro_f1=macro_f1, accuracy=accuracy)
            fold_rows.append(row)
            scores.append((macro_f1, accuracy))
            print(f'Candidate {candidate_id}/12 fold {fold}: macro F1={macro_f1:.6f}, accuracy={accuracy:.6f}', flush=True)
        values = np.array(scores)
        summary.append(dict(candidate_id=candidate_id, **params,
                            mean_macro_f1=float(values[:, 0].mean()), std_macro_f1=float(values[:, 0].std(ddof=0)),
                            mean_accuracy=float(values[:, 1].mean()), std_accuracy=float(values[:, 1].std(ddof=0))))
        pd.DataFrame(fold_rows).to_csv(args.output_dir / 'fold_scores.csv', index=False)
        pd.DataFrame(summary).to_csv(args.output_dir / 'candidate_scores.csv', index=False)
        print(f'Completed candidate {candidate_id}: {summary[-1]}', flush=True)
    best = max(summary, key=lambda row: row['mean_macro_f1'])
    best_params = {k: best[k] for k in SPACE}
    (args.output_dir / 'best_parameters.json').write_text(json.dumps({
        'candidate_id': best['candidate_id'], 'parameters': best_params,
        'fixed_parameters': fixed,
        'mean_cv_macro_f1': best['mean_macro_f1'], 'std_cv_macro_f1': best['std_macro_f1'],
        'mean_cv_accuracy': best['mean_accuracy'], 'std_cv_accuracy': best['std_accuracy'],
        'selection_metric': 'mean CV macro F1', 'final_test_evaluated': False,
        'full_development_model_fitted': False,
    }, indent=2) + '\n')
    print(f'Best candidate by mean CV macro F1: {best}', flush=True)


if __name__ == '__main__':
    main()
