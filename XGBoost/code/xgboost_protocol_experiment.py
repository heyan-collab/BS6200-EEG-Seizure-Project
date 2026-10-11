"""Add five-class protocol metrics and separate binary experiments.

Never generates splits. Each stage refuses existing output files.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, precision_recall_curve, auc)

from tune_xgboost import load_development
from xgboost_baseline import FEATURES, read_csv

MODEL_SEED = 42
GROUP_SPLIT_SEED = 6200  # User/group metadata; splits are read, never regenerated.
FIXED = dict(objective='binary:logistic', tree_method='hist', device='cpu',
             eval_metric='logloss', random_state=MODEL_SEED, n_jobs=4)
BASELINE = dict(n_estimators=100, max_depth=6, learning_rate=0.3,
                min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                reg_alpha=0.0, reg_lambda=1.0)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def empty_output(path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.mkdir(parents=True, exist_ok=True)


def hashes(paths):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def binary_metrics(y, prob):
    """Positive label 1 = seizure; threshold fixed before tuning at 0.5."""
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, prob, pos_label=1)
    return dict(n_samples=len(y), accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        seizure_precision=float(precision_score(y, pred, pos_label=1, zero_division=0)),
        seizure_recall_sensitivity=float(recall_score(y, pred, pos_label=1, zero_division=0)),
        specificity=float(tn / (tn + fp)),
        seizure_f1=float(f1_score(y, pred, pos_label=1, zero_division=0)),
        roc_auc=float(roc_auc_score(y, prob)),
        pr_auc=float(average_precision_score(y, prob, pos_label=1)),
        pr_auc_trapezoidal=float(auc(recall, precision)),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        confusion_matrix_class_order=[0, 1],
        classification_report=classification_report(y, pred, labels=[0, 1],
            target_names=['non_seizure_0', 'seizure_1'], output_dict=True, zero_division=0))


def preservation_manifest(root):
    """Capture pre-protocol files; new protocol paths are explicitly excluded."""
    new_sources = {'xgboost_protocol_experiment.py','report_xgboost_protocol.py','validate_xgboost_protocol.py'}
    new_docs = {'BINARY_XGBOOST_RESULTS.md','XGBOOST_PROTOCOL_COMPLIANCE.md','REFERENCES.md'}
    result = {}
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if '.git' in rel.parts or '__pycache__' in rel.parts or any(part.startswith('.venv') for part in rel.parts):
            continue
        if rel.parts[:2] == ('results','binary') or path.name == 'requirements-protocol.txt':
            continue
        if rel.parts[0] == 'src' and path.name in new_sources:
            continue
        if rel.parts[0] == 'docs' and path.name in new_docs:
            continue
        if rel.parts[:2] == ('results','final') and path.name.startswith('five_class_protocol_'):
            continue
        if rel.parts[:2] == ('results','figures') and (path.name.startswith('binary_') or path.name == 'five_class_protocol_roc.png'):
            continue
        result[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def preserve(root):
    path = root / 'results/binary/protected_original_files.json'
    if path.exists():
        raise FileExistsError('Preservation manifest already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, preservation_manifest(root))


def supplement(root):
    output = root / 'results/final/five_class_protocol_metrics.json'
    prob_path = root / 'results/final/five_class_protocol_test_probabilities.csv'
    if output.exists() or prob_path.exists():
        raise FileExistsError('Five-class supplemental outputs already exist')
    original = json.loads((root / 'results/final/metrics.json').read_text())
    saved = pd.read_csv(root / 'results/final/predictions.csv')
    split = read_csv(root / 'data/shared_train_test_split.csv')
    idx = split.loc[split.split == 'final_test', 'row_index'].to_numpy(dtype=int)
    model = xgb.Booster(); model.load_model(root / 'results/final/model.json')
    model.set_param({'nthread': 4})
    if model.feature_names != FEATURES or model.num_features() != 178:
        raise ValueError('Unexpected five-class feature mapping')
    data = read_csv(root / 'data/data.csv').iloc[idx]
    y = data.y.to_numpy(dtype=int)
    probabilities = model.predict(xgb.DMatrix(data[FEATURES]), strict_shape=True)
    if probabilities.shape != (2300, 5) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Invalid class probabilities')
    pred = saved.set_index('row_index').loc[idx, 'y_pred'].to_numpy(dtype=int)
    if not np.array_equal(probabilities.argmax(axis=1) + 1, pred):
        raise ValueError('Saved predictions disagree with loaded model')
    if not np.array_equal(y, saved.set_index('row_index').loc[idx, 'y_true'].to_numpy()):
        raise ValueError('Input labels disagree with saved final labels')
    results = dict(original['test'])  # Preserve existing scores and matrix exactly.
    results['balanced_accuracy'] = float(balanced_accuracy_score(y, pred))
    results['macro_roc_auc'] = float(roc_auc_score(y, probabilities, labels=[1, 2, 3, 4, 5], average='macro', multi_class='ovr'))
    results['roc_auc_convention'] = 'sklearn roc_auc_score: average=macro, multi_class=ovr, labels=[1,2,3,4,5]'
    results['probability_column_order'] = [1, 2, 3, 4, 5]
    results['metadata'] = dict(model_retrained=False, model_selection_performed=False,
        purpose='Complementary reporting only, after completed model selection',
        group_split_generation_seed=6200, model_seed=42,
        feature_set='F1 = Raw EEG, X1-X178 (CSV columns x1–x178)',
        input_sha256=hashes([root/'data/data.csv', root/'data/shared_train_test_split.csv', root/'results/final/model.json']))
    p = pd.DataFrame(probabilities, columns=[f'probability_class_{c}' for c in range(1, 6)])
    p.insert(0, 'row_index', idx); p.insert(1, 'y_true', y)
    p.to_csv(prob_path, index=False); write_json(output, results)
    print('Five-class Balanced Accuracy:', results['balanced_accuracy'], 'macro OVR ROC-AUC:', results['macro_roc_auc'], flush=True)


def tune(root):
    out = root / 'results/binary/tuning'; empty_output(out)
    X, original_encoded, folds, idx = load_development(root / 'data')
    y = (original_encoded == 0).astype(int)  # Original y=1 is positive.
    manifest = json.loads((root / 'results/tuning/search_manifest.json').read_text())
    candidates = manifest['candidates']  # Identical controlled 12-candidate scope.
    settings = dict(model_seed=42, search_seed=42, group_split_generation_seed=6200,
        selection_metric='mean five-fold seizure-class F1 (pos_label=1)',
        std_definition='population SD across five folds, ddof=0',
        strategy='Reuse the existing 12 ParameterSampler(seed=42) candidates; fresh binary models',
        parameter_space=manifest['search_space'], candidates=candidates,
        fixed_parameters=FIXED, positive_class='1 = seizure (original y=1)',
        negative_class='0 = non-seizure (original y=2,3,4,5)', threshold=0.5,
        feature_set='F1 = Raw EEG, X1-X178', final_test_used=False,
        input_sha256=hashes([root/'data/data.csv', root/'data/shared_train_test_split.csv', root/'data/shared_split_with_cv.csv']),
        versions={'xgboost': xgb.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__})
    write_json(out / 'search_manifest.json', settings)
    pd.DataFrame({'row_index': idx, 'cv_fold': folds}).to_csv(out/'development_cv_assignments.csv', index=False)
    scores, summaries = [], []
    for cid, params in enumerate(candidates, 1):
        per = []
        for fold in range(5):
            tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            model = xgb.XGBClassifier(**FIXED, **params)
            model.fit(X.iloc[tr], y[tr])
            prob = model.predict_proba(X.iloc[va])[:, 1]
            pred = (prob >= 0.5).astype(int)
            row = dict(candidate_id=cid, fold=fold, n_train=len(tr), n_validation=len(va),
                validation_seizure_support=int(y[va].sum()),
                seizure_f1=float(f1_score(y[va], pred, pos_label=1, zero_division=0)),
                accuracy=float(accuracy_score(y[va], pred)),
                balanced_accuracy=float(balanced_accuracy_score(y[va], pred)))
            scores.append(row); per.append(row)
            print(f'Binary candidate {cid}/12 fold {fold}: seizure F1={row["seizure_f1"]:.6f}', flush=True)
        summary = dict(candidate_id=cid, **params)
        for metric in ['seizure_f1', 'accuracy', 'balanced_accuracy']:
            v = np.array([r[metric] for r in per])
            summary['mean_'+metric] = float(v.mean()); summary['std_'+metric] = float(v.std(ddof=0))
        summaries.append(summary)
        pd.DataFrame(scores).to_csv(out/'fold_scores.csv', index=False)
        pd.DataFrame(summaries).to_csv(out/'candidate_scores.csv', index=False)
    best = max(summaries, key=lambda r: r['mean_seizure_f1'])  # Ties: candidate order.
    chosen = dict(candidate_id=best['candidate_id'], parameters={k: best[k] for k in candidates[0]},
        fixed_parameters=FIXED, selection_metric=settings['selection_metric'],
        threshold=0.5, final_test_used=False, positive_class=1)
    write_json(out/'best_parameters.json', chosen)
    write_json(out/'cv_summary.json', {**best, 'n_candidates':12, 'n_fits':60,
        'selection_metric':settings['selection_metric'], 'test_evaluated':False})
    print('Selected binary candidate:', best, flush=True)


def fit_evaluate(root, stage):
    out = root / f'results/binary/{stage}'; empty_output(out)
    X, original_encoded, folds, idx = load_development(root / 'data')
    y_train = (original_encoded == 0).astype(int)
    if stage == 'baseline':
        params = {**FIXED, **BASELINE}; selected = None
    else:
        chosen = json.loads((root/'results/binary/tuning/best_parameters.json').read_text())
        params = {**chosen['fixed_parameters'], **chosen['parameters']}; selected = chosen['candidate_id']
    model = xgb.XGBClassifier(**params)
    model.fit(X, y_train)  # One full-development fit for this stage.
    train_prob = model.predict_proba(X)[:, 1]
    # Final-test extraction happens only after this model is fully fitted.
    split = read_csv(root/'data/shared_train_test_split.csv')
    test_idx = split.loc[split.split == 'final_test', 'row_index'].to_numpy(dtype=int)
    data = read_csv(root/'data/data.csv')
    y_test = (data.iloc[test_idx].y.to_numpy() == 1).astype(int)
    test_prob = model.predict_proba(data.iloc[test_idx][FEATURES])[:, 1]  # Exactly one test prediction call.
    results = dict(training=binary_metrics(y_train, train_prob), test=binary_metrics(y_test, test_prob))
    results['metadata'] = dict(positive_class='Seizure = 1 (original y=1)',
        negative_class='Non-seizure = 0 (original y=2,3,4,5)', threshold=0.5,
        parameters=params, selected_candidate_id=selected, model_seed=42, search_seed=42,
        group_split_generation_seed=6200, feature_set='F1 = Raw EEG, X1-X178',
        pr_auc_convention='Average precision (sklearn average_precision_score); trapezoidal PR-AUC also saved separately',
        roc_auc_convention='Binary ROC-AUC from P(seizure=1)',
        fit_calls=1, final_test_prediction_calls=1, tuning_after_test=False,
        input_sha256=hashes([root/'data/data.csv',root/'data/shared_train_test_split.csv',root/'data/shared_split_with_cv.csv']),
        versions={'xgboost':xgb.__version__, 'sklearn':sklearn.__version__},
        limitation='Shared windows overlap in inferred recording groups; not unseen-recording or unseen-subject evaluation')
    model.get_booster().save_model(out/'model.json')
    predictions, probabilities = [], []
    for label, rows, true, prob in [('development',idx,y_train,train_prob),('final_test',test_idx,y_test,test_prob)]:
        predictions.append(pd.DataFrame({'row_index':rows,'column_a':data.iloc[rows].column_a.to_numpy(),'original_y':data.iloc[rows].y.to_numpy(),'split':label,'y_true':true,'y_pred':(prob>=0.5).astype(int)}))
        probabilities.append(pd.DataFrame({'row_index':rows,'split':label,'y_true':true,'probability_seizure_1':prob,'probability_non_seizure_0':1-prob}))
    pd.concat(predictions).sort_values('row_index').to_csv(out/'predictions.csv', index=False)
    pd.concat(probabilities).sort_values('row_index').to_csv(out/'probabilities.csv', index=False)
    for name in ['training','test']:
        matrix = pd.DataFrame(results[name]['confusion_matrix'],index=[0,1],columns=[0,1])
        matrix.index.name='true_class'; matrix.to_csv(out/f'confusion_matrix_{name}.csv')
    write_json(out/'metrics.json', results)
    if stage == 'final':
        gain = model.get_booster().get_score(importance_type='gain')
        importance = pd.DataFrame({'feature':FEATURES,'position':range(1,179),'gain':[float(gain.get(f,0)) for f in FEATURES]})
        importance = importance.sort_values(['gain','position'],ascending=[False,True]).reset_index(drop=True)
        importance.insert(0,'rank',range(1,179))
        importance.to_csv(out/'feature_importance_gain.csv',index=False)
    print(stage, json.dumps(results['test'],indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--stage',choices=['preserve','supplement','tune','baseline','final'],required=True)
    args=parser.parse_args()
    random.seed(42);np.random.seed(42)
    if args.stage=='preserve':preserve(args.root)
    elif args.stage=='supplement':supplement(args.root)
    elif args.stage=='tune':tune(args.root)
    else:fit_evaluate(args.root,args.stage)


if __name__=='__main__':main()
