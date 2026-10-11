"""Fit saved selected parameters once; score the shared final test set once."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier

from xgboost_baseline import FEATURES, CLASSES, read_csv, evaluate


def write_report(root, results, selected):
    """Render saved scores without making additional model predictions."""
    lines = ['# Final XGBoost results', '',
             'The final model was fitted once on all 9,200 development samples using',
             'the selected and fixed parameters from `results/tuning/best_parameters.json`.',
             'The original shared split is unchanged. Only `x1`–`x178` are features;',
             '`y` is the five-class target. Labels are encoded as 0–4 during fitting',
             'and restored to 1–5 for reports. Python, NumPy and model seeds are 42.', '',
             'No early stopping, test eval set, cross-validation or additional tuning',
             'was performed. The test set was scored in one prediction call after',
             'training. All metrics and this report use that same cached prediction.', '',
             '## Parameters', '', '```json',
             json.dumps({**selected['fixed_parameters'], **selected['parameters']}, indent=2),
             '```', '', '## Overall performance', '',
             '| Set | Samples | Accuracy | Macro precision | Macro recall | Macro F1 |',
             '|---|---:|---:|---:|---:|---:|']
    for name in ['training', 'test']:
        r = results[name]
        lines.append(f"| {name} | {r['n_samples']} | {r['accuracy']:.6f} | {r['macro_precision']:.6f} | {r['macro_recall']:.6f} | {r['macro_f1']:.6f} |")
    for name in ['training', 'test']:
        lines += ['', f'## {name.title()} per-class report', '',
                  '| Class | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
        for label in ['1', '2', '3', '4', '5', 'macro avg', 'weighted avg']:
            r = results[name]['classification_report'][label]
            lines.append(f"| {label} | {r['precision']:.6f} | {r['recall']:.6f} | {r['f1-score']:.6f} | {int(r['support'])} |")
        lines += ['', 'Confusion matrix: rows are true classes, columns are predicted',
                  'classes, both ordered 1–5.', '', '```text']
        lines += [' '.join(f'{n:4d}' for n in row) for row in results[name]['confusion_matrix']]
        lines += ['```']
    lines += ['', '## Comparison with baseline test results', '',
              'Differences are final minus baseline, in percentage points.', '',
              '| Metric | Baseline | Final | Difference (pp) |', '|---|---:|---:|---:|']
    for metric, r in results['baseline_test_comparison'].items():
        lines.append(f"| {metric} | {r['baseline']:.6f} | {r['final']:.6f} | {r['difference_percentage_points']:+.4f} |")
    lines += ['', '## Files and reproduction', '',
              '- `src/train_final_xgboost.py`: validates inputs and split, loads saved',
              '  parameters, fits once, predicts each set once, saves outputs and report.',
              '- `results/final/model.json`: native XGBoost booster model.',
              '- `results/final/predictions.csv`: original row index, sample ID, split,',
              '  true label and predicted label for training and test samples.',
              '- `results/final/metrics.json`: full reports, baseline comparison,',
              '  parameter settings, package versions, file hashes and evaluation count.',
              '- `results/final/confusion_matrix_training.csv` and',
              '  `results/final/confusion_matrix_test.csv`: matrices with labeled axes.', '',
              'Install `requirements.txt`, then run from the repository root:', '',
              '```sh', 'python src/train_final_xgboost.py', '```', '',
              'The script refuses existing nonempty output directories or an existing',
              'report so that accidental reruns cannot overwrite this experiment.',
              'The current experiment is complete; another run is unnecessary.', '',
              '## Interpretation limits', '',
              'Training metrics are in-sample. Test metrics describe the provided',
              'window-level split, which has inferred recording-group overlap:',
              'all 2,300 test windows have same-group windows in training. Therefore',
              'these results do not establish generalization to unseen recordings or',
              'subjects. Baseline and final scores use the same test samples; differences',
              'are descriptive, without a statistical significance claim. The baseline',
              'previously scored this test set; "once" here refers to the final model.',
              'No further tuning was performed after this final evaluation. Existing',
              'baseline and tuning files remain unchanged.', '']
    root.write_text('\n'.join(lines))


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--best-parameters', type=Path, default=root / 'results/tuning/best_parameters.json')
    parser.add_argument('--baseline-metrics', type=Path, default=root / 'results/baseline/metrics.json')
    parser.add_argument('--output-dir', type=Path, default=root / 'results/final')
    parser.add_argument('--report', type=Path, default=root / 'docs/FINAL_RESULTS.md')
    args = parser.parse_args()
    if (args.output_dir.exists() and any(args.output_dir.iterdir())) or args.report.exists():
        raise FileExistsError('Final outputs/report already exist; refusing to rerun or overwrite')
    selected = json.loads(args.best_parameters.read_text())
    params = {**selected['fixed_parameters'], **selected['parameters']}
    if set(selected['fixed_parameters']) & set(selected['parameters']):
        raise ValueError('Overlapping saved parameter fields')
    if params['random_state'] != 42:
        raise ValueError('Expected saved seed 42')
    random.seed(42)
    np.random.seed(42)
    data = read_csv(args.data_dir / 'data.csv')
    split = read_csv(args.data_dir / 'shared_train_test_split.csv')
    idx = pd.to_numeric(split['row_index'], errors='raise')
    if idx.isna().any() or not np.isfinite(idx).all() or (idx % 1 != 0).any():
        raise ValueError('Invalid row_index')
    if len(idx) != len(data) or idx.duplicated().any() or set(idx) != set(range(len(data))):
        raise ValueError('Split must cover every row exactly once')
    if set(split['split']) != {'development', 'final_test'}:
        raise ValueError('Unexpected split labels')
    train_idx = idx[split['split'] == 'development'].to_numpy(dtype=int)
    test_idx = idx[split['split'] == 'final_test'].to_numpy(dtype=int)
    if len(train_idx) != 9200 or len(test_idx) != 2300:
        raise ValueError('Unexpected shared split sizes')
    X_train = data.iloc[train_idx][FEATURES].apply(pd.to_numeric, errors='raise')
    y_train = pd.to_numeric(data.iloc[train_idx]['y'], errors='raise').to_numpy()
    if not np.isfinite(X_train.to_numpy()).all() or set(y_train) != set(CLASSES):
        raise ValueError('Invalid development data')
    # The only fit uses development data; saved settings are not modified.
    model = XGBClassifier(**params)
    model.fit(X_train, y_train.astype(int) - 1)
    train_pred = model.predict(X_train).astype(int) + 1
    # Access final-test features/labels only after fitting. Exactly one test prediction.
    X_test = data.iloc[test_idx][FEATURES].apply(pd.to_numeric, errors='raise')
    y_test = pd.to_numeric(data.iloc[test_idx]['y'], errors='raise').to_numpy()
    if not np.isfinite(X_test.to_numpy()).all() or not set(y_test).issubset(CLASSES):
        raise ValueError('Invalid test data')
    test_pred = model.predict(X_test).astype(int) + 1
    results = {'training': evaluate(y_train, train_pred), 'test': evaluate(y_test, test_pred)}
    baseline = json.loads(args.baseline_metrics.read_text())
    for name, h in baseline['metadata']['input_sha256'].items():
        if hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() != h:
            raise ValueError('Baseline and final input bytes differ')
    results['baseline_test_comparison'] = {
        k: {'baseline': baseline['test'][k], 'final': results['test'][k],
            'difference': results['test'][k] - baseline['test'][k],
            'difference_percentage_points': 100 * (results['test'][k] - baseline['test'][k])}
        for k in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    }
    inputs = [args.data_dir / 'data.csv', args.data_dir / 'shared_train_test_split.csv', args.best_parameters, args.baseline_metrics]
    results['metadata'] = {
        'parameters': params, 'seed': 42, 'features': FEATURES, 'class_order': CLASSES,
        'selected_candidate_id': selected['candidate_id'], 'fit_calls': 1,
        'final_test_prediction_calls': 1, 'further_tuning_performed': False,
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'limitation': 'Shared split has inferred recording-group overlap; not an unseen-recording or unseen-subject evaluation.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(args.output_dir / 'model.json')
    frames = []
    for name, rows, true, pred in [('development', train_idx, y_train, train_pred), ('final_test', test_idx, y_test, test_pred)]:
        frames.append(pd.DataFrame({'row_index': rows, 'column_a': data.iloc[rows]['column_a'].to_numpy(), 'split': name, 'y_true': true, 'y_pred': pred}))
    pd.concat(frames).sort_values('row_index').to_csv(args.output_dir / 'predictions.csv', index=False)
    (args.output_dir / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    for name in ['training', 'test']:
        matrix = pd.DataFrame(results[name]['confusion_matrix'], index=CLASSES, columns=CLASSES)
        matrix.index.name = 'true_class'
        matrix.columns.name = 'predicted_class'
        matrix.to_csv(args.output_dir / f'confusion_matrix_{name}.csv')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report, results, selected)
    print(json.dumps({k: v for k, v in results.items() if k != 'metadata'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
