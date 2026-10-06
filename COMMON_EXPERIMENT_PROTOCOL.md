# BS6200 Common Experimental Protocol

## 1. Tasks

### Binary classification
- Seizure: original label y = 1
- Non-seizure: original labels y = 2, 3, 4, 5

### Five-class classification
- Use original labels y = 1, 2, 3, 4, 5

## 2. Data Split
- Development set: 80%
- Final test set: 20%
- Random seed: 6200
- All members must use the same shared split.

## 3. Cross-Validation
- 5-fold stratified cross-validation on the development set.
- If reliable recording/group IDs can be recovered, group-aware CV should be considered.

## 4. Common Feature Sets
- F1: Raw EEG, X1-X178
- F2: Time-domain features
- F3: Frequency-domain features
- F4: Combined time + frequency features

Deep learning models may use raw EEG directly if handcrafted features are not appropriate.

## 5. Binary Classification Metrics
- Accuracy
- Balanced Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Primary metric: Seizure-class F1-score

## 6. Five-Class Metrics
- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Macro ROC-AUC
- Confusion Matrix

Primary metric: Macro F1

## 7. Leakage Prevention
Any operation that learns from data must be fitted only on the training portion:
- Scaling
- Imputation
- PCA
- Feature selection
- Resampling / SMOTE
- Threshold tuning

The final test set must not be used for tuning.

## 8. Required Outputs for Each Model
Each member should provide:
- Model code
- Reference paper / code
- Hyperparameters
- CV results
- Final test results
- Confusion matrix
- Binary results
- Five-class results
- Model interpretation
- Error analysis
- Main strengths and limitations

## 9. Shared Rule
Different models, same experimental protocol.
