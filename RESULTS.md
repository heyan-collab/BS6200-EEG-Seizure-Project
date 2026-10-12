# TCN experiment results

## Selected model

- Task: binary seizure recognition (`y == 1` versus `y in {2, 3, 4, 5}`)
- Input: raw ordered EEG values `X1` to `X178`, standardised with the mean and SD of each training fold
- Validation: the five fixed development folds in `shared_split_with_cv.csv`
- Model: temporal convolutional network (PyTorch), 5 residual blocks, dilations 1–16, kernel size 5, dropout 0.2
- Channels per layer: 32
- Receptive field: 249 samples (covers the full 178-sample segment)
- Class weight: none
- Optimiser: Adam, learning rate 0.001, batch size 64, up to 60 epochs, early stopping patience 8
- Early stopping: on a stratified 10% split of the training folds (the validation fold is never used for training or model selection within a fold)
- Random seed: 6200
- Primary selection metric: seizure-class F1
- Final test accessed: yes, once, after the configuration was locked (`run_tcn_final_predictions.py`)

## Final five-fold cross-validation (development set)

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.9912 | 0.0042 |
| Balanced accuracy | 0.9829 | 0.0075 |
| Precision | 0.9867 | 0.0095 |
| Recall | 0.9690 | 0.0131 |
| Specificity | 0.9967 | 0.0023 |
| F1 | **0.9778** | **0.0107** |
| AUROC | 0.9992 | 0.0005 |
| Average precision | 0.9973 | 0.0018 |

The fold-level F1 scores were 0.9850, 0.9905, 0.9793, 0.9643, 0.9699. The standard deviation across folds was 0.0107.

![ROC and PR curves](outputs/cv_selection/figures/binary_roc_pr_cv.png)

![Fold-level metrics](outputs/cv_selection/figures/binary_fold_metrics.png)

![Binary confusion matrix](outputs/cv_selection/figures/binary_confusion_matrix.png)

Pooled over the five validation folds, 1783 of 1840 seizure segments were detected (57 missed), and 24 of 7360 non-seizure segments were falsely flagged as seizures.

![Training curves](outputs/cv_selection/figures/binary_training_curves.png)

Selected epochs per fold: 27, 25, 23, 18, 16. Training the five folds took 33.4 minutes on `cpu`.

## Class-weight ablation (32 channels)

| Configuration | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|
| No class weight | **0.9778** | **0.9867** | 0.9690 | **0.9967** |
| Balanced class weight | 0.9632 | 0.9477 | **0.9799** | 0.9861 |

Relative to no weighting, balanced weighting changed recall by +0.0109 and precision by -0.0390, giving F1 = 0.9632 versus 0.9778. Because F1 is the predefined primary metric, no class weighting was selected.

![Class-weight ablation](outputs/cv_selection/figures/ablation_class_weight.png)

## Channel-count study (no class weight)

| Channels | F1 | Accuracy | AUROC | F1 standard deviation | Minutes |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.9726 | 0.9892 | **0.9993** | 0.0109 | 34.2 |
| 32 | **0.9778** | **0.9912** | 0.9992 | **0.0107** | 33.4 |
| 64 | 0.9750 | 0.9901 | 0.9991 | 0.0124 | 71.0 |

The 32-channel configuration achieved the highest mean F1 (0.9778) and was retained as the final binary TCN.

![Channel study](outputs/cv_selection/figures/channel_study.png)

## Final five-class cross-validation

Five-class model: same architecture, 32 channels, cross-entropy loss (classes are balanced, so no class weight).

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.8016 | 0.0131 |
| Balanced accuracy | 0.8016 | 0.0131 |
| Macro precision | 0.8048 | 0.0135 |
| Macro recall | 0.8016 | 0.0131 |
| Macro F1 | **0.8009** | **0.0139** |
| Macro ROC-AUC (one-vs-rest) | 0.9624 | 0.0045 |

The fold-level Macro F1 scores were 0.8151, 0.8079, 0.7866, 0.7854, 0.8095.

![Five-class fold metrics](outputs/cv_selection/figures/multiclass_fold_metrics.png)

### Five-class channel-count study

| Channels | Accuracy | Macro F1 | Macro ROC-AUC | Macro F1 standard deviation |
|---:|---:|---:|---:|---:|
| 16 | 0.7304 | 0.7201 | 0.9425 | 0.0369 |
| 32 | 0.8016 | **0.8009** | 0.9624 | 0.0139 |
| 64 | **0.8040** | 0.8007 | **0.9645** | **0.0133** |

## Five-class error analysis (cross-validation)

Per-class scores are computed on the pooled out-of-fold predictions (each segment predicted by the model that did not train on it).

| Original class | Recording condition | Precision | Recall | F1 |
|---:|---|---:|---:|---:|
| 1 | Seizure activity | 0.9835 | 0.9707 | **0.9770** |
| 2 | Tumour region | 0.6482 | 0.6168 | 0.6321 |
| 3 | Healthy brain region of a patient with a tumour | 0.6349 | 0.6587 | 0.6466 |
| 4 | Eyes closed | 0.9218 | 0.8712 | 0.8958 |
| 5 | Eyes open | 0.8257 | 0.8908 | 0.8570 |

The most frequent errors were class 2 predicted as class 3 (616 segments), class 3 predicted as class 2 (553 segments), class 4 predicted as class 5 (222 segments), class 5 predicted as class 4 (114 segments). Seizure activity (class 1) achieved F1 = 0.9770.

![Five-class confusion matrix](outputs/cv_selection/figures/multiclass_confusion_matrix.png)

![Example signals](outputs/cv_selection/figures/example_signals.png)

## Final test (official, 2,300 segments)

The locked configuration (32 channels, no class weight) was trained as five fold models on the fixed development folds and their final-test probabilities were averaged (`run_tcn_final_predictions.py`). The final test was evaluated once and not used for any tuning.

### Binary

| Metric | CV mean | CV SD | Final test |
|---|---:|---:|---:|
| Accuracy | 0.9912 | 0.0042 | 0.9913 |
| Balanced accuracy | 0.9829 | 0.0075 | 0.9856 |
| Precision | 0.9867 | 0.0095 | 0.9803 |
| Recall | 0.9690 | 0.0131 | 0.9761 |
| Specificity | 0.9967 | 0.0023 | 0.9951 |
| F1 | **0.9778** | **0.0107** | **0.9782** |
| ROC-AUC | 0.9992 | 0.0005 | 0.9996 |
| PR-AUC | 0.9973 | 0.0018 | 0.9985 |

CV-test gap in F1: 0.0004. On the final test, 11 of 460 seizure segments were missed and 9 of 1,840 non-seizure segments were falsely flagged.

![Binary final-test confusion matrix](outputs/binary_final_test/confusion_matrix.png)

![Binary final-test ROC curve](outputs/binary_final_test/roc_curve.png)

![Binary final-test precision-recall curve](outputs/binary_final_test/precision_recall_curve.png)

### Five-class

| Metric | CV mean | CV SD | Final test |
|---|---:|---:|---:|
| Accuracy | 0.8016 | 0.0131 | 0.7943 |
| Balanced accuracy | 0.8016 | 0.0131 | 0.7943 |
| Macro precision | 0.8048 | 0.0135 | 0.7943 |
| Macro recall | 0.8016 | 0.0131 | 0.7943 |
| Macro F1 | **0.8009** | **0.0139** | **0.7940** |
| Macro ROC-AUC | 0.9624 | 0.0045 | 0.9618 |

CV-test gap in Macro F1: 0.0068.

| Original class | Recording condition | Precision | Recall | F1 | ROC-AUC (OvR) |
|---:|---|---:|---:|---:|---:|
| 1 | Seizure activity | 0.9825 | 0.9783 | 0.9804 | 0.9994 |
| 2 | Tumour region | 0.6283 | 0.6652 | 0.6463 | 0.9210 |
| 3 | Healthy brain region of a patient with a tumour | 0.6282 | 0.5913 | 0.6092 | 0.9172 |
| 4 | Eyes closed | 0.8875 | 0.9087 | 0.8980 | 0.9915 |
| 5 | Eyes open | 0.8448 | 0.8283 | 0.8364 | 0.9801 |

The most frequent final-test errors were class 3 predicted as class 2 (163 segments), class 2 predicted as class 3 (136), class 5 predicted as class 4 (49) and class 4 predicted as class 5 (38), the same pairs that dominated cross-validation. Seizure activity remained highly separable (F1 0.9804).

![Five-class final-test confusion matrix](outputs/five_class_final_test/confusion_matrix.png)

Selected epochs in the final-test run: binary 20, 35, 34, 19, 36; five-class 28, 21, 30, 38, 34.

## Model interpretation, strengths, and limitations

The TCN stacks dilated causal 1-D convolutions with residual connections. Each block doubles the dilation, so a small number of layers covers the whole segment (receptive field 249 samples). The convolution filters are learned end-to-end from the raw signal together with the classifier.

Strengths: learned temporal filters, no hand-crafted time- or frequency-domain features, a compact model, and direct use of raw ordered EEG samples. Limitations: results depend on stochastic training (initialisation, batch order, early stopping), learned filters are not directly interpretable clinically, and the supplied CSV contains no reliable subject or recording identifiers, so the shared split is stratified at segment level rather than subject level. This limitation must be disclosed.

## Reproducibility notes

- Cross-validation and selection results come from `tcn_seizure_en.py` (files in `outputs/cv_selection/`, one row per fold). Standardisation statistics, early-stopping splits and model fitting are computed inside each training fold only. Standard deviations are sample SDs (ddof = 1) across the five folds.
- Final-test results come from `run_tcn_final_predictions.py`, which retrains the same locked configuration on the same five folds and averages the five fold models' probabilities. CV and final-test numbers are therefore from two independent training runs of an identical configuration; the CV numbers were not recomputed in the final-test run.
- Confusion matrices, per-class metrics and final-test curves are derived from the saved predictions by `make_final_test_figures.py`.
- Training is stochastic: an independent retraining of the same configuration gave identical binary final-test metrics and a five-class final-test Macro F1 about 0.02 different, which should be read as run-to-run variability.
