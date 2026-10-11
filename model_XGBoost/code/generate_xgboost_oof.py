"""Generate two development-only OOF files with fixed selected XGBoost models.

Binary probability is P(seizure=1); five-class probability columns follow
original Classes 1–5. Per-fold scores must reproduce saved supplementary CV
metrics before either file is written. Pooled nonlinear metrics need not equal
mean fold metrics. Existing outputs/models are never written or loaded.
"""
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score)

from xgboost_baseline import FEATURES, read_csv
from xgboost_protocol_experiment import binary_metrics


def metrics(task, y, pred, prob):
    if task == 'binary':
        m = binary_metrics(y, prob)
        return {k: m['seizure_recall_sensitivity' if k=='seizure_recall' else k]
                for k in ['accuracy','balanced_accuracy','seizure_precision',
                          'seizure_recall','specificity','seizure_f1','roc_auc','pr_auc']}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[1,2,3,4,5], average='macro', zero_division=0)
    return dict(accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        macro_precision=float(precision),macro_recall=float(recall),macro_f1=float(f1),
        macro_roc_auc=float(roc_auc_score(y,prob,labels=[1,2,3,4,5],average='macro',multi_class='ovr')))


def main():
    root = Path(__file__).resolve().parent.parent
    destinations = {'binary': root/'results/binary/tuning/oof_predictions.csv',
                    'five_class': root/'results/tuning/oof_predictions.csv'}
    if any(p.exists() for p in destinations.values()):
        raise FileExistsError('OOF outputs already exist; refusing to overwrite')
    tracked = subprocess.check_output(['git','-C',str(root),'ls-files'],text=True).splitlines()
    protected = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in tracked}
    random.seed(42); np.random.seed(42)
    split = read_csv(root/'data/shared_train_test_split.csv').set_index('row_index').sort_index()
    cv = read_csv(root/'data/shared_split_with_cv.csv').set_index('row_index').sort_index()
    if list(split.index)!=list(range(11500)) or list(cv.index)!=list(range(11500)) or not split.split.equals(cv.split):
        raise ValueError('Invalid shared split/index coverage')
    idx = split.index[split.split=='development'].to_numpy(dtype=int)
    test = set(split.index[split.split=='final_test'])
    membership = set(idx)
    if len(idx)!=9200 or len(membership)!=9200 or membership & test:
        raise ValueError('Invalid development membership')
    # Only development records are loaded; final-test records are skipped.
    data = pd.read_csv(root/'data/data.csv',encoding='utf-8-sig',
        skiprows=lambda row: row>0 and row-1 not in membership)
    data.index = idx
    X = data[FEATURES]; original_y = data.y.to_numpy(dtype=int)
    folds = cv.loc[idx,'cv_fold'].to_numpy()
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(folds).all() or set(folds)!=set(range(5)):
        raise ValueError('Invalid development features or folds')
    pending = {}
    for task,path in destinations.items():
        directory = path.parent
        settings = json.loads((directory/'best_parameters.json').read_text())
        params = {**settings['fixed_parameters'],**settings['parameters']}
        if params['random_state']!=42:raise ValueError('Expected model seed 42')
        assignment = pd.read_csv(directory/'development_cv_assignments.csv')
        if not np.array_equal(idx,assignment.row_index) or not np.array_equal(folds,assignment.cv_fold):
            raise ValueError('Development assignments differ from completed experiment')
        expected = pd.read_csv(directory/'selected_model_full_fold_metrics.csv').sort_values('fold')
        summary = json.loads((directory/'selected_model_full_cv_summary.json').read_text())
        if summary['metadata']['parameters']!=params or summary['metadata']['std_ddof']!=0:
            raise ValueError('Saved metric settings differ from selected configuration')
        y = (original_y==1).astype(int) if task=='binary' else original_y
        records = []
        for fold in range(5):
            train = np.flatnonzero(folds!=fold); valid = np.flatnonzero(folds==fold)
            if len(train)!=7360 or len(valid)!=1840 or set(idx[train]) & set(idx[valid]):
                raise ValueError('Invalid held-out fold')
            if set(idx[train]) & test or set(idx[valid]) & test:
                raise ValueError('Final-test contamination')
            model = xgb.XGBClassifier(**params)
            model.fit(X.iloc[train], y[train] if task=='binary' else y[train]-1)
            probability = model.predict_proba(X.iloc[valid])
            if task=='binary':
                if not np.array_equal(model.classes_,[0,1]):raise ValueError('Unexpected binary class order')
                probability = probability[:,1]
                prediction = (probability>=settings['threshold']).astype(int)
            else:
                if not np.array_equal(model.classes_,range(5)):raise ValueError('Unexpected multiclass order')
                if probability.shape!=(1840,5) or not np.allclose(probability.sum(axis=1),1,atol=1e-6):
                    raise ValueError('Invalid multiclass probabilities')
                prediction = probability.argmax(axis=1)+1
            record = pd.DataFrame({'row_index':idx[valid],'fold':fold,'y_true':y[valid],'y_pred':prediction})
            if task=='binary':record['probability']=probability
            else:
                for c in range(1,6):record[f'probability_class_{c}']=probability[:,c-1]
            score = metrics(task,y[valid],prediction,probability)
            target = expected[expected.fold==fold]
            if len(target)!=1 or any(not np.isclose(v,target.iloc[0][k],rtol=0,atol=1e-12) for k,v in score.items()):
                raise RuntimeError(f'{task} fold {fold} does not reproduce existing CV; nothing saved')
            records.append(record)
            print(f'{task} fold {fold}: held-out membership verified; all saved fold metrics reproduced.',flush=True)
        oof = pd.concat(records).sort_values('row_index').reset_index(drop=True)
        if len(oof)!=9200 or oof.row_index.nunique()!=9200 or set(oof.row_index)!=membership or set(oof.row_index)&test:
            raise RuntimeError('OOF sample coverage failed; nothing saved')
        if not np.array_equal(oof.fold.to_numpy(),cv.loc[oof.row_index,'cv_fold'].to_numpy()):
            raise RuntimeError('OOF fold mapping failed; nothing saved')
        # Verify again from the assembled OOF table, not transient predictions.
        per_fold = []
        for fold,group in oof.groupby('fold',sort=True):
            prob = group.probability.to_numpy() if task=='binary' else group[[f'probability_class_{c}' for c in range(1,6)]].to_numpy()
            per_fold.append(metrics(task,group.y_true.to_numpy(),group.y_pred.to_numpy(),prob))
        scores = pd.DataFrame(per_fold)
        for key,stat in summary['metrics'].items():
            if not np.isclose(scores[key].mean(),stat['mean'],rtol=0,atol=1e-12) or not np.isclose(scores[key].std(ddof=0),stat['sd'],rtol=0,atol=1e-12):
                raise RuntimeError(f'{task} {key} summary mismatch; nothing saved')
        prob = oof.probability.to_numpy() if task=='binary' else oof[[f'probability_class_{c}' for c in range(1,6)]].to_numpy()
        pooled = metrics(task,oof.y_true.to_numpy(),oof.y_pred.to_numpy(),prob)
        print(task+' pooled OOF metrics: '+json.dumps(pooled),flush=True)
        print(task+' fold-mean metrics: '+json.dumps(scores.mean().to_dict()),flush=True)
        pending[path] = oof
    for path,h in protected.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=h:
            raise RuntimeError(f'Existing file changed: {path}; nothing saved')
    for path,oof in pending.items():oof.to_csv(path,index=False)
    print('PASS: saved exactly two OOF CSV files; all previous files unchanged.',flush=True)


if __name__=='__main__':main()
