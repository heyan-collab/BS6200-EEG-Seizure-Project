# MultiRocket experiment results

## Selected model

- Task: binary seizure recognition (`y == 1` versus `y in {2, 3, 4, 5}`)
- Input: raw ordered EEG values `X1` to `X178`
- Validation: the five fixed development folds in `shared_split_with_cv.csv`
- Model: `aeon.classification.convolution_based.MultiRocketClassifier`
- Kernels: 10,000
- Class weight: none
- Random seed: 6200
- Primary selection metric: seizure-class F1
- Final test accessed: no

## Final five-fold cross-validation

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.9941 | 0.0006 |
| Balanced accuracy | 0.9878 | 0.0018 |
| Precision | 0.9934 | 0.0031 |
| Recall | 0.9772 | 0.0041 |
| Specificity | 0.9984 | 0.0008 |
| F1 | **0.9852** | **0.0015** |
| AUROC | 0.9995 | 0.0007 |
| Average precision | 0.9990 | 0.0008 |

The fold-level F1 scores were 0.9836, 0.9864, 0.9835, 0.9863, and 0.9863. The small standard deviation indicates stable performance across the predefined folds.

## Class-weight ablation (1,000 kernels)

| Configuration | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|
| No class weight | **0.9828** | **0.9864** | 0.9793 | **0.9966** |
| Balanced class weight | 0.9764 | 0.9666 | **0.9864** | 0.9914 |

Balanced weighting increased recall but reduced precision enough to lower the predefined primary metric. Therefore, no class weighting was selected.

## Kernel-count study (no class weight)

| Kernels | F1 | Accuracy | AUROC | F1 standard deviation |
|---:|---:|---:|---:|---:|
| 1,000 | 0.9828 | 0.9932 | 0.9993 | 0.0020 |
| 10,000 | **0.9852** | **0.9941** | **0.9995** | **0.0015** |

Increasing the kernel count produced a modest F1 improvement and slightly better fold-to-fold stability. The 10,000-kernel configuration was therefore retained as the final MultiRocket model.

## Reproducibility notes

The reported values come from the CSV files under `outputs/binary_cv5_k10000_final`. Preprocessing and model fitting are performed independently inside each training fold. The locked final-test partition was not read during these experiments.
