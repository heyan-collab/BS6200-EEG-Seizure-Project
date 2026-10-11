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
- Final test accessed: yes, once after freezing the configuration

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

## Binary final-test evaluation

| Metric | Final test |
|---|---:|
| Accuracy | 0.9939 |
| Balanced accuracy | 0.9864 |
| Precision | 0.9956 |
| Recall / sensitivity | 0.9739 |
| Specificity | 0.9989 |
| F1 | **0.9846** |
| AUROC | 0.9995 |
| Average precision / PR-AUC | 0.9986 |

The final-test confusion counts were TN = 1,838, FP = 2, FN = 12, and TP = 448. Final-test F1 was only 0.0006 below the five-fold CV mean, indicating close agreement between CV and held-out performance.

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

## Final five-class cross-validation

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.8502 | 0.0069 |
| Balanced accuracy | 0.8502 | 0.0069 |
| Macro precision | 0.8510 | 0.0063 |
| Macro recall | 0.8502 | 0.0069 |
| Macro F1 | **0.8501** | **0.0067** |
| Macro ROC-AUC (one-vs-rest) | 0.9675 | 0.0024 |

The fold-level Macro F1 scores were 0.8512, 0.8605, 0.8500, 0.8457, and 0.8431.

## Five-class final-test evaluation

| Metric | Final test |
|---|---:|
| Accuracy | 0.8443 |
| Balanced accuracy | 0.8443 |
| Macro precision | 0.8445 |
| Macro recall | 0.8443 |
| Macro F1 | **0.8442** |
| Macro ROC-AUC (one-vs-rest) | 0.9688 |

| Original class | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 1 | 0.9891 | 0.9826 | **0.9858** |
| 2 | 0.7927 | 0.7565 | 0.7742 |
| 3 | 0.7609 | 0.7609 | 0.7609 |
| 4 | 0.8755 | 0.8717 | 0.8736 |
| 5 | 0.8045 | 0.8500 | 0.8266 |

The main final-test errors were class 2 predicted as class 3 (92 segments), class 3 predicted as class 2 (82), class 4 predicted as class 5 (57), and class 5 predicted as class 4 (50). The final-test Macro F1 was 0.0059 below the CV mean and reproduced the same dominant confusion pairs.

### Five-class kernel-count study

| Kernels | Accuracy | Macro F1 | Macro ROC-AUC | Macro F1 standard deviation |
|---:|---:|---:|---:|---:|
| 1,000 | 0.8189 | 0.8183 | 0.9580 | 0.0088 |
| 10,000 | **0.8502** | **0.8501** | **0.9675** | **0.0067** |

The larger transform improved Macro F1 by 0.0318 and reduced fold-to-fold variation.

## Five-class error analysis

| Original class | Recording condition | Precision | Recall | F1 |
|---:|---|---:|---:|---:|
| 1 | Seizure activity | 0.9874 | 0.9826 | **0.9850** |
| 2 | Tumour region | 0.7770 | 0.7538 | 0.7652 |
| 3 | Healthy brain region of a patient with a tumour | 0.7700 | 0.7625 | 0.7662 |
| 4 | Eyes closed | 0.9041 | 0.8663 | 0.8848 |
| 5 | Eyes open | 0.8154 | 0.8859 | 0.8492 |

The dominant errors were class 2 predicted as class 3 (365 segments), class 3 predicted as class 2 (349), class 4 predicted as class 5 (229), and class 5 predicted as class 4 (149). These pairs represent closely related recording conditions, so their overlap is plausible. In contrast, seizure activity remained highly separable: class 1 achieved F1 = 0.9850.

## Model interpretation, strengths, and limitations

MultiRocket applies many fixed convolutional kernels across multiple dilations and summarizes the resulting response patterns before fitting a regularized linear classifier. It therefore captures short and long temporal motifs without manually designed time- or frequency-domain features.

Strengths observed here are high seizure-class performance, low fold-to-fold variance, direct use of raw ordered EEG samples, and substantially faster training than many end-to-end deep networks. Limitations are weak clinical interpretability of individual random kernels, reduced discrimination between physiologically similar non-seizure classes, and the absence of reliable subject or recording-group identifiers in the supplied CSV. Consequently, the shared split is stratified at segment level rather than subject level, and this limitation must be disclosed.

## Reproducibility notes

CV values come from the selected 10,000-kernel output folders. Preprocessing and model fitting were performed independently inside each training fold. After the configuration was frozen, separate models were trained on all 9,200 development samples and evaluated once on the 2,300-sample final test set. Final-test metrics use the base MultiRocket predictions. The probability files use logistic calibration fitted only on development OOF decision scores; the final test was not used to fit the calibrator.
