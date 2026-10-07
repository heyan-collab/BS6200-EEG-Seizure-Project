# BS6200 EEG Seizure Project — InceptionTime

This project trains and evaluates an InceptionTime-style model on raw EEG (X1–X178), using the group's fixed development/test split and five validation folds. Both binary and five-class experiments are complete.

## Start here

- **Main code:** [inceptiontime_sharedsplit.py](inceptiontime_sharedsplit.py). This single file includes training, five-fold validation, final testing, and result export for both tasks.
- **Main results:** [inceptiontime_summary.csv](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/inceptiontime_summary.csv). It contains six rows: training, validation mean, and test for binary and five-class classification. Validation standard deviations are in the `_val_sd` columns.
- **Dependencies:** [requirements.txt](requirements.txt).
- **Fixed partitions:** [shared_split_with_cv.csv](splits/shared_split_with_cv.csv).
- **Dataset:** [Epileptic Seizure Recognition.csv](<Epileptic Seizure Recognition.csv>).

## Tasks and shared protocol

| Task | Labels | Primary metric |
| --- | --- | --- |
| Binary | Original y=1 → seizure (target 1); y=2,3,4,5 → non-seizure (target 0) | Seizure F1 |
| Five-class | Original labels 1–5; internal targets 0–4 in the same order | Macro F1 |

The supplied split has **9,200 development samples (80%)** and **2,300 test samples (20%)**, with seed **6200**. Each of the five development folds uses 7,360 training rows and 1,840 validation rows. Both tasks use exactly the same uploaded assignments.

Imputation and normalization are fitted only on each fold's training rows. Final preprocessing is fitted on the full development set. The test set is excluded from early stopping and tuning. Binary prediction uses a fixed 0.5 seizure threshold (ties select non-seizure). The model uses F1 raw EEG; it does not redefine the shared handcrafted feature sets.

## Which output file should I read?

All links below point to files currently uploaded under `results_inceptiontime_sharedsplit/cpu_10epochs_20261007/`.

| Purpose | Binary files | Five-class files |
| --- | --- | --- |
| Validation metrics: one row per fold | [cv_fold_metrics.csv](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/binary/cv_fold_metrics.csv) | [cv_fold_metrics.csv](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/five_class/cv_fold_metrics.csv) |
| Validation predictions: each development row predicted by its held-out fold model | [cv_predictions.csv](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/binary/cv_predictions.csv) | [cv_predictions.csv](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/five_class/cv_predictions.csv) |
| Validation confusion matrix: pooled out-of-fold predictions | [cv_confusion_matrix.png](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/binary/cv_confusion_matrix.png) | [cv_confusion_matrix.png](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/five_class/cv_confusion_matrix.png) |
| Training loss by epoch (not classification metrics) | [final_training_history.csv](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/binary/final_training_history.csv) | [final_training_history.csv](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/five_class/final_training_history.csv) |
| Training loss plot | [final_training_history.png](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/binary/final_training_history.png) | [final_training_history.png](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/five_class/final_training_history.png) |
| Final test predictions and probabilities | [final_test_predictions.csv](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/binary/final_test_predictions.csv) | [final_test_predictions.csv](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/five_class/final_test_predictions.csv) |
| Final test confusion matrix | [final_test_confusion_matrix.png](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/binary/final_test_confusion_matrix.png) | [final_test_confusion_matrix.png](results_inceptiontime_sharedsplit/cpu_10epochs_20261007/five_class/final_test_confusion_matrix.png) |

**Training, validation, and test mean different things:**

- **Training:** the final model evaluated on the same 9,200 development rows used to fit it; these scores are in-sample.
- **Validation:** five held-out fold scores, summarized as mean ± sample standard deviation. CV also selects the stopping epoch.
- **Test:** the final model evaluated on the fixed 2,300 test rows after training.

Use `inceptiontime_summary.csv` for all three sets of classification metrics. A training-history CSV records loss over epochs; it is not a table of accuracy or F1. Validation confusion matrices pool predictions, while the reported validation metrics average the five fold scores.

Five-class prediction CSV targets and decisions use 0–4, corresponding to original labels 1–5. The `original_label` column retains the original labels. Binary target 1 means seizure.

## Results

All values below are proportions from 0 to 1. Validation is mean ± standard deviation across five folds.

### Binary classification

| Metric | Training | Validation mean ± SD | Test |
| --- | --- | --- | --- |
| Accuracy | 0.9907 | 0.9909 ± 0.0026 | 0.9883 |
| Balanced accuracy | 0.9913 | 0.9841 ± 0.0062 | 0.9878 |
| Precision | 0.9621 | 0.9815 ± 0.0115 | 0.9558 |
| Recall / sensitivity | 0.9924 | 0.9728 ± 0.0136 | 0.9870 |
| Seizure F1 | 0.9770 | 0.9771 ± 0.0066 | 0.9711 |
| ROC-AUC | 0.9996 | 0.9992 ± 0.0006 | 0.9992 |
| Specificity | 0.9902 | 0.9954 ± 0.0029 | 0.9886 |
| PR-AUC (average precision) | 0.9986 | 0.9971 ± 0.0019 | 0.9972 |

### Five-class classification

| Metric | Training | Validation mean ± SD | Test |
| --- | --- | --- | --- |
| Accuracy | 0.8030 | 0.8012 ± 0.0068 | 0.7922 |
| Balanced accuracy | 0.8030 | 0.8012 ± 0.0068 | 0.7922 |
| Macro precision | 0.8128 | 0.8035 ± 0.0071 | 0.8015 |
| Macro recall | 0.8030 | 0.8012 ± 0.0068 | 0.7922 |
| Macro F1 | 0.8039 | 0.8001 ± 0.0044 | 0.7933 |
| Macro ROC-AUC (one-vs-rest) | 0.9660 | 0.9626 ± 0.0007 | 0.9604 |

## Model and training settings

This is a smaller **single-network adaptation** of InceptionTime, rather than the paper's five-network ensemble. It has six Inception modules, kernels 40/20/10, eight filters per branch, bottleneck width eight, residual shortcuts every three modules, global average pooling, and a two- or five-class output layer.

| Setting | Value |
| --- | --- |
| Input | Raw EEG, one channel × 178 timepoints |
| Preprocessing | Training-only per-timepoint median imputation; one global training-amplitude mean and standard deviation |
| Optimizer / learning rate | Adam / 0.001 |
| Batch size | 128 |
| Maximum CV epochs / patience | 10 / 3 |
| Early stopping | Validation seizure F1 or macro F1 |
| Loss | Unweighted cross entropy; no resampling |
| Final training epochs | Binary: 6; five-class: 10 (median best epoch across folds) |
| Seed / device | 6200 / CPU, two threads |

## Reproduce training, validation, and testing

From the repository root, with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python inceptiontime_sharedsplit.py --task both --epochs 10
```

The script reads `Epileptic Seizure Recognition.csv` and `splits/shared_split_with_cv.csv` by default. A rerun writes a new timestamped directory under `results_inceptiontime_sharedsplit/`. Use `--task binary` or `--task five_class` to run one task. The saved results can be read without rerunning training; numerical reproducibility can vary across hardware and library versions.

Additional checkpoint and JSON files in the uploaded folders support reloading and detailed inspection. The files linked above are the main files needed to understand the experiment.

## Interpretation, errors, and limitations

Parallel convolution kernels learn EEG patterns at several temporal scales. Exploratory temporal occlusion found the largest average predicted-probability drop when masking X91–X108. This is a sensitivity observation, not a causal or physiological explanation.

Binary testing produced **27 errors**, including **6 missed seizures**. Five-class testing produced **478 errors**; label 2 → 3 (149 cases) and label 3 → 2 (162 cases) accounted for most of them.

**Possible recording leakage:** EEG ID suffixes suggest 500 candidate recording groups. Of these, 496 cross development/test boundaries and all 500 cross CV folds. Recording provenance is unconfirmed. The uploaded sample-level split was retained for group comparison as requested; these results do not demonstrate independent-recording or independent-patient generalization.

Strengths include learning directly from raw EEG, multiple temporal scales, and consistent shared partitions. Limitations include the smaller single network, short training budget, limited interpretability, and possible recording leakage. Test errors were not used to retune the model.

## Reference

Fawaz et al. (2020), *InceptionTime: Finding AlexNet for Time Series Classification*. [Paper](https://doi.org/10.1007/s10618-020-00710-y) · [Reference code](https://github.com/hfawaz/InceptionTime).
