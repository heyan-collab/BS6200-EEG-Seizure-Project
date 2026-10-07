# BS6200 EEG Seizure Project — CNN-BiLSTM

This branch contains the **CNN-BiLSTM implementation, fixed shared-split training pipeline, and experiment outputs** for the BS6200 EEG Seizure Recognition group project.

The model is evaluated under the same group protocol used for model comparison:

- the same development / final-test split,
- the same fixed 5-fold cross-validation assignments,
- the same label definitions,
- leakage-safe preprocessing,
- common evaluation metrics.

> Previous setup instructions are preserved in `README_original.md`.  
> A Chinese explanation for personal review is available in `CNN_BiLSTM_Model_Note_ZH.md`.

---

## 1. Model overview

**Model:** 1D CNN + Bidirectional LSTM (CNN-BiLSTM)

**Input:** one EEG segment with 178 sequential measurements.

**Tasks:**

1. Binary seizure detection  
   - Seizure: original label `y = 1`
   - Non-seizure: original labels `y = 2, 3, 4, 5`

2. Five-class EEG state classification  
   - Original labels `y = 1, 2, 3, 4, 5`

The main idea is:

```text
Raw EEG
   ↓
1D CNN
   ↓
Local waveform representation
   ↓
Bidirectional LSTM
   ↓
Temporal-context representation
   ↓
Classifier
```

The CNN learns local EEG waveform patterns automatically, while the BiLSTM models how these learned patterns are ordered across the short EEG segment.

---

## 2. Project files

```text
cnn-bilstm branch
│
├── README.md
├── README_original.md
├── CNN_BiLSTM_Model_Note_ZH.md
│
├── bs6200_cnn_bilstm.py
├── cnn_bilstm_sharedsplit_v3.py
│
├── splits/
│   └── shared_split_with_cv.csv
│
├── results_cnn_bilstm_quick/
│
└── results_cnn_bilstm_sharedsplit/
    └── full_25epochs_<timestamp>/
```

### Important files

- `bs6200_cnn_bilstm.py`  
  Original reference implementation.

- `cnn_bilstm_sharedsplit_v3.py`  
  Current formal implementation using the group-wide fixed split and CV assignments.

- `splits/shared_split_with_cv.csv`  
  Shared sample index file used by all group models.

- `results_cnn_bilstm_quick/`  
  Earlier smoke-test outputs.

- `results_cnn_bilstm_sharedsplit/`  
  Formal runs using the shared split and fixed CV folds.

---

## 3. Shared experimental protocol

The formal V3 pipeline reads:

```text
splits/shared_split_with_cv.csv
```

Current split:

- Total samples: **11,500**
- Development set: **9,200**
- Final test set: **2,300**
- Fixed CV folds: **0, 1, 2, 3, 4**
- Random seed: **6200**

For each CV fold:

```text
7,360 samples → training
1,840 samples → validation
2,300 final-test samples → untouched
```

The script does **not** regenerate train/test splits or CV folds.

This is essential for fair comparison between CNN-BiLSTM and other models such as MultiRocket, InceptionTime, XGBoost, or other deep-learning approaches.

---

## 4. Leakage prevention

Standardization is performed inside each training fold.

```text
Training fold
   ↓
Fit StandardScaler
   ↓
Transform training fold
   ↓
Transform validation fold using the same scaler
```

The validation fold is never used to estimate scaling parameters.

For the final model:

```text
All development samples
   ↓
Fit final scaler
   ↓
Transform development set
   ↓
Transform final test set using the same scaler
```

The final test set therefore remains isolated from model development.

---

## 5. CNN-BiLSTM architecture

```text
178-point EEG
    ↓
Conv1D: 1 → 32 channels, kernel size 5
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Max Pooling
    ↓
Conv1D: 32 → 64 channels, kernel size 5
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Max Pooling
    ↓
Conv1D: 64 → 128 channels, kernel size 3
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Bidirectional LSTM
hidden size = 64 per direction
    ↓
Fully connected: 128 → 64
    ↓
ReLU
    ↓
Dropout = 0.30
    ↓
Output layer
```

### CNN

The 1D CNN learns local waveform patterns such as:

- abrupt signal changes,
- short oscillatory structures,
- local amplitude patterns,
- combinations of neighboring EEG values.

The features are learned directly from raw EEG rather than manually defined.

### Batch Normalization

Batch Normalization stabilizes intermediate activations and can make optimization more consistent.

### ReLU

ReLU introduces nonlinearity:

```text
ReLU(x) = max(0, x)
```

This allows the network to learn nonlinear EEG patterns.

### Max Pooling

Max Pooling reduces sequence length while preserving strong local responses.

Approximately:

```text
178 → 89 → 44 time steps
```

This lowers computational cost before the BiLSTM stage.

### BiLSTM

The Bidirectional LSTM reads the learned sequence in two directions:

```text
Forward:  beginning → end
Backward: end → beginning
```

This allows the model to use both earlier and later context within a complete EEG segment.

Because this is offline segment classification, using both directions does not introduce test-information leakage.

---

## 6. CNN-LSTM vs. CNN-BiLSTM

A standard CNN-LSTM uses a one-directional LSTM.

A CNN-BiLSTM uses both forward and backward recurrent processing.

```text
CNN-LSTM:
CNN features → forward LSTM → classification

CNN-BiLSTM:
CNN features → forward LSTM
             → backward LSTM
             → combined representation
             → classification
```

BiLSTM may capture richer context, but it also introduces more parameters.

Because each EEG sample contains only 178 points, whether bidirectional recurrence provides a real benefit must be determined experimentally.

---

## 7. Feature representation

This model uses:

```text
Raw EEG: X1–X178
```

No explicit handcrafted feature engineering is used.

There is also no explicit feature-selection step such as:

- SelectKBest,
- L1 feature selection,
- PCA,
- SHAP-based selection,
- permutation-importance selection.

Instead, the CNN performs **automatic representation learning**.

This distinction is important when comparing models:

```text
Feature-based model:
Raw EEG
   ↓
Handcrafted features
   ↓
Optional feature selection
   ↓
Classifier

CNN-BiLSTM:
Raw EEG
   ↓
CNN learned representation
   ↓
BiLSTM
   ↓
Classifier
```

---

## 8. Class imbalance

The binary task contains more non-seizure than seizure samples.

The current pipeline uses **class-weighted cross-entropy** by default.

Class weights are calculated from the training fold only.

This increases the contribution of the minority seizure class during optimization without generating synthetic EEG samples.

---

## 9. Five-fold cross-validation

The development set is evaluated using the fixed five folds defined in the shared split file.

For each fold:

```text
4 folds → training
1 fold  → validation
```

Each development sample serves as validation exactly once.

The model therefore reports:

```text
mean ± standard deviation
```

The mean measures average performance, while the standard deviation reflects stability across folds.

---

## 10. Early stopping and final epoch selection

The formal run allows a maximum of **25 epochs per CV fold**.

Early stopping monitors validation loss.

Observed best epochs for the current binary experiment:

```text
[9, 19, 25, 10, 16]
```

The median is:

```text
16 epochs
```

The final model is therefore retrained for 16 epochs on the entire development set.

This training duration is selected using development-set CV only, not the final test set.

---

## 11. Final model training

After five-fold CV:

```text
Five CV best epochs
   ↓
Median = 16
   ↓
Fit scaler on all 9,200 development samples
   ↓
Train a new CNN-BiLSTM for 16 epochs
   ↓
Evaluate once on the 2,300-sample final test set
```

The final test set is not used for architecture selection, preprocessing selection, or epoch selection.

---

## 12. Evaluation metrics

### Binary task

Reported metrics:

- Accuracy
- Balanced Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

**Primary metric:** seizure-class F1-score.

### Five-class task

Common group metrics:

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Macro ROC-AUC
- Confusion Matrix

**Primary metric:** Macro F1.

---

## 13. Current binary results

### Five-fold CV

| Metric | Mean ± SD |
| --- | ---: |
| Accuracy | **0.9884 ± 0.0049** |
| Balanced Accuracy | **0.9870 ± 0.0051** |
| Precision | **0.9584 ± 0.0182** |
| Recall / Sensitivity | **0.9848 ± 0.0056** |
| Specificity | **0.9893 ± 0.0048** |
| F1-score | **0.9714 ± 0.0118** |
| ROC-AUC | **0.9992 ± 0.0004** |
| PR-AUC | **0.9969 ± 0.0016** |

### Final test

| Metric | Result |
| --- | ---: |
| Accuracy | **0.9857** |
| Balanced Accuracy | **0.9878** |
| Precision | **0.9402** |
| Recall / Sensitivity | **0.9913** |
| Specificity | **0.9842** |
| F1-score | **0.9651** |
| ROC-AUC | **0.9993** |
| PR-AUC | **0.9974** |

Final-test confusion matrix:

```text
                     Predicted
                 Non-seizure   Seizure
True Non-seizure      1811         29
True Seizure             4        456
```

Therefore:

- True negatives: **1811**
- False positives: **29**
- False negatives: **4**
- True positives: **456**

The current model has very high seizure sensitivity and misses only four seizure samples in this final test set.

---

## 14. Training-curve interpretation

The final training loss decreases strongly across training, with small local fluctuations.

These fluctuations are normal in mini-batch stochastic optimization.

The final-model training plot contains only training loss because the final model is trained on the full development set.

Validation-based model selection has already been performed during five-fold CV.

Therefore, overfitting should be assessed mainly from:

- fold-specific validation losses,
- early stopping behavior,
- cross-validation performance,
- stability across folds.

---

## 15. Current strengths

The main strengths of this model are:

- direct learning from raw EEG,
- automatic local waveform representation learning,
- temporal-context modeling with BiLSTM,
- strong seizure sensitivity,
- fixed shared split and CV folds,
- leakage-safe scaling,
- class-weighted training,
- consistent CV performance.

---

## 16. Current limitations

The current result does **not** prove that:

- BiLSTM is necessary,
- CNN-BiLSTM is better than MultiRocket or other model families,
- binary performance implies equally strong five-class performance,
- the model generalizes to unseen patients or recording sessions,
- learned patterns are necessarily clinically meaningful seizure physiology.

A key dataset limitation is that these are short pre-segmented EEG windows.

If multiple segments from the same original recording appear in both training and test sets, segment-level splitting may produce optimistic performance estimates.

This limitation should remain explicit unless reliable recording-level grouping can be reconstructed.

---

## 17. Protocol components still to be added

The current V3 core already implements:

- shared train/test split,
- fixed five-fold CV,
- leakage-safe standardization,
- class weighting,
- early stopping,
- common evaluation metrics,
- final-test isolation.

Additional project-level analyses still planned include:

- repeated runs with multiple random seeds,
- hyperparameter tuning / nested CV if computationally feasible,
- threshold analysis using development-set out-of-fold predictions,
- false-positive / false-negative error analysis,
- five-class result interpretation,
- direct comparison with other model families.

---

## 18. Standard fields for cross-model comparison

| Field | CNN-BiLSTM |
| --- | --- |
| Input representation | Raw EEG, 178 points |
| Handcrafted features | No |
| Automatic feature learning | Yes, 1D CNN |
| Explicit feature selection | No |
| Temporal modeling | Bidirectional LSTM |
| Imbalance handling | Class-weighted loss |
| Shared train/test split | Yes |
| Shared fixed CV folds | Yes |
| Leakage-safe preprocessing | Yes |
| Binary primary metric | Seizure-class F1 |
| Five-class primary metric | Macro F1 |
| Interpretability | Limited |
| Main strength | Local waveform learning + bidirectional temporal context |
| Main limitation | Higher complexity, limited interpretability, possible segment-level split optimism |

The same table should be completed for all other project models.

---

## 19. Output files

Typical formal run:

```text
results_cnn_bilstm_sharedsplit/
└── full_25epochs_<timestamp>/
    ├── run_config.json
    ├── all_results.json
    │
    ├── binary/
    │   ├── cv_fold_metrics.csv
    │   ├── results_summary.json
    │   ├── final_test_predictions.csv
    │   ├── final_test_confusion_matrix.png
    │   ├── final_training_history.csv
    │   ├── final_training_history.png
    │   ├── final_model.pt
    │   └── fold_0 ... fold_4/
    │
    └── five/
        └── corresponding five-class outputs
```

### File meanings

`cv_fold_metrics.csv`  
Stores one row per CV fold and is the key file for cross-model mean ± SD comparison.

`results_summary.json`  
Stores CV summaries, best epochs, and final-test metrics.

`final_test_predictions.csv`  
Stores sample-level predictions and probabilities. This is useful for later threshold analysis, error analysis, and ensemble construction.

`final_test_confusion_matrix.png`  
Visualizes false positives and false negatives.

`final_training_history.csv / .png`  
Records final-model training loss.

`final_model.pt`  
Stores the final PyTorch model state and scaling parameters.

---

## 20. Run commands

### Quick smoke test

```bash
python cnn_bilstm_sharedsplit_v3.py --task both --quick
```

### Formal run

```bash
python cnn_bilstm_sharedsplit_v3.py --task both --epochs 25
```

The default paths are:

```text
data(1).csv
splits/shared_split_with_cv.csv
```

---

## 21. One-sentence presentation summary

**CNN-BiLSTM learns local EEG waveform patterns with 1D convolutions and then models their bidirectional temporal context with a BiLSTM, while using the same fixed split and cross-validation folds as the other BS6200 models.**

---

## 22. Current status

- **Binary task:** formal result completed.
- **Five-class task:** formal result to be added after completion.
- **Next step:** compare this model with other approaches using the same split, CV folds, and evaluation metrics.
