# XGBoost common-protocol compliance

Authority: BS6200 Common Experimental Protocol supplied by the user in this task. The standalone COMMON_EXPERIMENT_PROTOCOL.md is not present in this repository; no shared protocol file was created or modified.

## Seeds and features

**Group split-generation seed: 6200. Model/search/SHAP seeds: 42.** Shared assignments were consumed unchanged; no new split was generated, and the generation seed is recorded from group/user specification rather than reconstructed from assignments.

**Feature set: F1 = Raw EEG, X1-X178**, represented by x1–x178 in the CSV. F2–F4 are not used or independently redefined; shared src/common/features.py is therefore not needed for this F1 experiment.

## Protocol checklist

| Protocol item | Five-class | Binary | Evidence / limitation |
|---|---|---|---|
| 1. Target task | Original labels 1–5 | Seizure 1; non-seizure 0 | Binary 1 iff original y=1 |
| 2. Shared 80/20 split, seed 6200 | Preserved | Preserved | 9,200/2,300; exact shared index membership |
| 3. Shared stratified five-fold development CV | Existing folds | Same existing folds | Original five-class and binary class balance verified; no generation |
| 3. Reliable group IDs require group-aware splits | Unresolved group-level condition | Same unresolved condition | Inferred groups overlap; provenance not confirmed; no unilateral change |
| 4. Common feature definitions | F1 raw EEG | F1 raw EEG | No F2/F3/F4 redefinition |
| 5. Binary metrics and seizure-F1 selection | Not applicable | All nine present | Accuracy, Balanced Accuracy, seizure precision/recall/F1, specificity, ROC-AUC, PR-AUC, matrix |
| 6. Five-class metrics and macro-F1 selection | All seven present | Not applicable | Original final metrics plus complementary Balanced Accuracy and macro OVR ROC-AUC |
| 7. Leakage prevention | Development-only tuning | Development-only tuning | No learned preprocessing or threshold tuning; threshold 0.5 fixed |
| 8. Model code and reference | Present | Present | Existing code; protocol stages; REFERENCES.md |
| 8. Hyperparameters and CV results | Present | Present | Existing tuning outputs; results/binary/tuning |
| 8. Final test results and confusion matrix | Present | Present | Original and supplemental five-class outputs; results/binary/final |
| 8. Both target tasks | Complete | Complete | Original five-class model preserved; new independent binary model |
| 8. Model interpretation | Existing gain and SHAP | Binary gain added | Attribution is not causal/clinical importance |
| 8. Error analysis, strengths and limitations | Existing final reports plus notes below | BINARY_XGBOOST_RESULTS.md | Window-level claims only |
| 9. Same protocol across models | Shared F1/membership/folds | Same | Group must resolve recording-ID and PR-AUC conventions consistently |

## Complementary five-class metrics

Balanced Accuracy: **0.701304**. Macro ROC-AUC: **0.924832**, computed by sklearn roc_auc_score with average='macro', multi_class='ovr', labels=[1,2,3,4,5], using saved-model probabilities in that class order. No retraining, retuning or selection used this result.

| Metric | Existing final test result |
|---|---:|
| accuracy | 0.701304 |
| macro_precision | 0.703851 |
| macro_recall | 0.701304 |
| macro_f1 | 0.700904 |

The original final confusion matrix and classification report are copied into the complementary metrics JSON without alteration. New probabilities have a distinct filename, five_class_protocol_test_probabilities.csv. This is an additional post-selection probability pass for missing AUC reporting, not a new model selection exercise. The final test set was already evaluated previously; it is not newly unseen.

Five-class error analysis: Classes 2 and 3 are often confused (165 class-2 windows predicted as 3; 114 class-3 windows predicted as 2), and 109 class-4 windows are predicted as class 5. Class 1 has F1 0.936543; Class 2 has F1 0.555160.

Five-class strengths: complete reproducible nonlinear raw-EEG pipeline, shared comparisons, macro-F1 selection and gain/SHAP interpretation. Limitations: training metrics are 1.0 versus test accuracy 0.701304; class-specific errors differ; source overlap and CV selection optimism prevent independent-recording claims. Macro ROC-AUC measures ranking and does not replace the observed hard-label metrics or imply clinical usefulness.

## Status and unresolved group decisions

All required **F1 XGBoost output categories and metrics** for both tasks are present. This does not certify the group-aware splitting condition: source/subject identifiers have not been authenticated and the current shared split has inferred recording overlap. Reliable IDs would require a group-agreed group-aware train/test and CV design under the protocol. No replacement was generated here.

PR-AUC is explicitly average precision, with trapezoidal PR-AUC also saved. The protocol does not specify which convention; all group members should report and compare the same convention.

Original five-class baseline, tuning, final, visualization, gain and SHAP files were preserved byte-for-byte. Added numerical files, figures and reports are new. Verification: src/validate_xgboost_protocol.py, with results/binary/protocol_validation.json.

## Reference

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785
