# BS6200 CNN-BiLSTM Reference Pipeline

This folder contains a runnable PyTorch reference implementation adapted to the
BS6200 common experimental protocol for the UCI Epileptic Seizure Recognition
dataset.

## Expected data

Place these in the same folder:

- `bs6200_cnn_bilstm.py`
- `data(1).csv`

The CSV must contain:

- `column_a`
- `x1` ... `x178`
- `y`

## Install

In Anaconda Prompt:

```bash
conda activate YOUR_ENVIRONMENT
pip install numpy pandas scikit-learn matplotlib torch
```

If you already have PyTorch installed, you do not need to reinstall it.

## First run: smoke test

Run this first to check that your environment and data loading work:

```bash
python bs6200_cnn_bilstm.py --data "data(1).csv" --task both --quick
```

Quick mode uses 2-fold CV and 3 epochs. It is NOT the final experiment.

## Full experiment

```bash
python bs6200_cnn_bilstm.py --data "data(1).csv" --task both --folds 5 --epochs 30
```

The script automatically uses CUDA if PyTorch detects a GPU.

## Outputs

`results_cnn_bilstm/`

- `shared_train_test_split.csv`
- `all_results.json`
- `binary/`
  - `cv_fold_metrics.csv`
  - `results_summary.json`
  - `final_test_predictions.csv`
  - `confusion_matrix.png`
  - `training_history.png`
  - `final_model.pt`
- `five/`
  - same output structure

## Important methodological choices

1. A single shared 80/20 split is created using the original five-class labels.
2. The final 20% test set is not used during cross-validation.
3. Five-fold stratified CV is performed only on the development set.
4. StandardScaler is fitted inside each training fold only.
5. Binary task:
   - y=1 -> seizure
   - y=2,3,4,5 -> non-seizure
6. Five-class task keeps all original five classes.
7. Class weights are computed from each training fold.
8. Binary primary metric: seizure-class F1.
9. Five-class primary metric: Macro F1.

## Important limitation

The CSV is a preprocessed/restructured dataset containing 1-second EEG segments.
If reliable original-recording IDs cannot be reconstructed, the current
stratified split is segment-level rather than recording-level. This should be
reported as a limitation because segments originating from the same original
recording may be correlated.

## Reference status

This is an adaptation, not an exact reproduction of the IEEE paper.
The model follows the CNN-BiLSTM idea and is adapted to the BS6200 unified
evaluation protocol and the actual 178-point CSV structure.
