# InceptionTime for EEG Seizure Classification

## Overview

This folder contains an **InceptionTime** implementation for the BS6200 Machine Learning group project. The model classifies EEG time-series segments using a shared experimental protocol, enabling comparison with other group models.

Two prediction tasks are studied:

- **Binary:** seizure (`original y = 1` → `1`) versus non-seizure (`original y = 2, 3, 4, 5` → `0`).
- **Five-class:** original labels `1`–`5`, encoded as `0`–`4` for model training.

## Dataset

- **Dataset:** [Epileptic Seizure Recognition (Kaggle)](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition)
- **Samples:** 11,500
- **Inputs:** 178 ordered EEG values (`X1`–`X178`) per sample
- **Classes:** Five equally sized original classes (2,300 samples each)
- **Model input shape (aeon):** `(n_samples, 1, 178)` — one channel and 178 time points

The original dataset's `Unnamed` column is not an EEG input feature. The shared split file identifies samples by `row_index`; preserve the original row order when joining.

## Experimental Protocol

| Item | Setting |
| --- | --- |
| Development split | 9,200 samples (80%) |
| Held-out final test | 2,300 samples (20%) |
| Cross-validation | 5 predefined stratified folds on development only |
| Split file | `shared_split_with_cv.csv` |
| Random seed | `6200` |
| Selected training epochs | `20` (selected based on CV performance) |
| Batch size | `64` |

A fresh model is trained for each CV fold. After choosing the configuration from development-set CV, a fresh model is trained on **all 9,200 development samples** and evaluated on the **untouched 2,300-sample final test set**. The test set must not be used for tuning.

## Model

InceptionTime uses convolutional inception-style modules to learn time-series features at multiple temporal scales. This project uses `aeon.classification.deep_learning.InceptionTimeClassifier`, trained from scratch rather than a pretrained checkpoint.

The configuration used in the experiments is:

```python
InceptionTimeClassifier(
    n_classifiers=1,
    n_epochs=20,
    batch_size=64,
    depth=6,
    n_filters=32,
    kernel_size=40,
    random_state=6200,
    verbose=False,
)
```


## Evaluation

| Metric | Binary task | Five-class task |
| --- | --- | --- |
| Accuracy | Overall | Overall |
| Balanced accuracy | Yes | Yes |
| Precision | Seizure class | Macro average |
| Recall | Seizure sensitivity | Macro average |
| F1 | **Seizure-class F1 (primary)** | **Macro F1 (primary)** |
| Specificity | Yes | Not reported |
| ROC-AUC | Seizure-class ROC-AUC | Macro one-vs-rest ROC-AUC |
| PR-AUC | Average precision (AP) | Not reported |
| Confusion matrix | 2 × 2 | 5 × 5 |

**Note:** The code's `pr_auc` field uses `sklearn.metrics.average_precision_score`, which computes **average precision**, not trapezoidal area under the precision–recall curve. State this distinction in any report.

## Outputs

The cross-validation workflow writes the following files (subject to the output directory configured in the script):

```text
inceptiontime_results_20epochs_complete/
├── inceptiontime_all_cv_results.csv
├── inceptiontime_cv_summary.csv
├── binary_cv_probabilities.csv
├── five_class_cv_probabilities.csv
├── binary_confusion_matrix.csv
├── binary_confusion_matrix.png
├── five_class_confusion_matrix.csv
└── five_class_confusion_matrix.png
```

The final test workflow writes:

```text
inceptiontime_final_results_20epochs/
├── inceptiontime_final_test_results.csv
├── binary_test_probabilities.csv
├── five_class_test_probabilities.csv
├── binary_test_confusion_matrix.csv
├── binary_test_confusion_matrix.png
├── five_class_test_confusion_matrix.csv
└── five_class_test_confusion_matrix.png
```

Prediction files include `row_index` so that predictions can be aligned with other group members' models for ensemble analysis.

## Running the Project

1. Install a compatible Python environment and required packages:

   ```bash
   python -m pip install numpy pandas matplotlib scikit-learn aeon tensorflow kagglehub
   ```

2. Obtain the EEG CSV from Kaggle and the group's `shared_split_with_cv.csv` file.
3. Update input and output paths for your machine. **Google Colab `/content/...` paths will not work directly in local VS Code.**
4. Run the data preparation and model-definition code before running the CV or final test code.
5. Run cross-validation to obtain validation metrics and out-of-fold probabilities.
6. Run final training/testing using the CV-selected configuration.

> **Note:** Exact commands and package versions depend on how the code is organized in this repository. Update this section once the final `.py` filenames and tested environment are fixed.

## Reproducibility and Limitations

- Use the group's provided split and fold assignments, rather than generating new random splits.
- Preserve consistent label mappings across CV, final testing, and ensemble prediction files.
- Do not include the held-out test set in CV, feature fitting, or hyperparameter selection.
- The tabular EEG samples may derive from longer original recordings. Without reliable recording-level identifiers, sample-level splits may not completely prevent recording-level leakage; interpret generalization claims accordingly.
- GPU and CPU runs may differ slightly because of computational nondeterminism.

## References

1. **Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., Webb, G. I., Idoumghar, L., Muller, P.-A., & Petitjean, F. (2020).** InceptionTime: Finding AlexNet for time series classification. *Data Mining and Knowledge Discovery, 34*, 1936–1962. https://doi.org/10.1007/s10618-020-00710-y
2. **InceptionTime original implementation:** https://github.com/hfawaz/InceptionTime
3. **aeon documentation — InceptionTimeClassifier:** https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.classification.deep_learning.InceptionTimeClassifier.html
4. **Dataset:** https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition

