# MultiRocket model

This folder implements MultiRocket for the shared BS6200 EEG experiment.

## Environment

```bash
conda activate bs6200-multirocket
python -m pip install -r models/multirocket/requirements.txt
```

## Input

- Raw features: `X1` to `X178`
- Binary target: original `y == 1` is seizure; `y in {2,3,4,5}` is non-seizure
- Fixed split and folds: `splits/shared_split_with_cv.csv`
- Expected aeon shape: `(n_samples, 1, 178)`

The raw EEG CSV is intentionally not committed. Pass its local path with `--data`.

## First smoke test

Run from the repository root:

```bash
python models/multirocket/train_multirocket.py --data "D:\留学\sem1\BS6200\6200-group project\Epileptic Seizure Recognition.csv" --task binary --fold 0 --n-kernels 100 --smoke-test
```

The smoke test uses 400 training and 200 validation samples. Its results are only a software check and must not be reported as the official experiment.

## Preliminary full-fold run

After the smoke test succeeds, run one complete fold with 1,000 kernels:

```bash
python models/multirocket/train_multirocket.py --data "D:\留学\sem1\BS6200\6200-group project\Epileptic Seizure Recognition.csv" --task binary --fold 0 --n-kernels 1000
```

Do not run all five folds or the final test until the group confirms the official kernel budget and compute-time plan.

## Outputs

Each run writes:

- `fold_metrics.csv`
- `cv_summary.csv`
- `oof_predictions.csv`
- `run_config.json`
- confusion matrix
- ROC curve and precision-recall curve for the binary task

Continuous Ridge decision scores are used for AUROC and average precision. Aeon's fallback `predict_proba` is not used because the default Ridge classifier does not provide calibrated probabilities.

## Leakage control

- The fixed `cv_fold` column defines validation membership.
- MultiRocket is fitted from scratch inside each training fold.
- The final test set is not accessed by this script.
- Threshold tuning and model selection must not use final-test results.
