# BS6200 six-model deep comparison — latest data

## Source priority
The latest workbook `BS6200_Shared_Model_Comparison_Template_prefilled3(1).xlsx` is the official source for aggregate model metrics.
The latest model archives are used for sample-level analyses.

## Numbers to replace from the previous visualization set
- MultiRocket five-class Final Macro F1: **0.8442**, not 0.8569.
- XGBoost five-class Final Macro F1: **0.7009**, not 0.7107.
- XGBoost five-class Final Accuracy: **0.7013**, not 0.7113.
- Previous per-class / agreement / hardness / overlap figures should be replaced by this package.

## Official final-test ranking
Binary: InceptionTime 0.9857 > MultiRocket 0.9846 > TCN 0.9782 > CNN-Transformer 0.9751 > CNN-BiLSTM 0.9651 > XGBoost 0.9272.

Five-class: MultiRocket 0.8442 > InceptionTime 0.8062 > TCN 0.7940 > CNN-Transformer 0.7805 > CNN-BiLSTM 0.7496 > XGBoost 0.7009.

## Sample-level difficulty
- Binary: 2198/2300 (95.57%) samples were correct for all six; 3 were wrong for all six.
- Five-class: 1071/2300 (46.57%) samples were correct for all six; 43 were wrong for all six.

## Class-wise result
Class 2 is the most difficult class for most models. MultiRocket is comparatively balanced across the five classes. CNN-BiLSTM is especially strong on Class 5 but weak on Class 2.

## Exploratory ensemble screen
- Best binary post-hoc equal-weight combination: TCN + MultiRocket + InceptionTime + CNN-Transformer + XGBoost — F1 0.9934.
- Best five-class post-hoc equal-weight combination: TCN + MultiRocket — Macro F1 0.8657.

These ensemble rankings are exploratory only because the combinations were ranked on the final test set. Under the shared protocol, a final ensemble must be selected on CV/OOF predictions first and then evaluated once on the held-out final test set.

## Workbook items to confirm, not modify
Because the latest Excel is authoritative, these values are preserved:
- CNN-Transformer Binary `CV F1 Mean = 0.000886`, identical to its `CV Specificity SD`.
- CNN-Transformer Five-class `CV Macro Recall Mean = 0.010583`, identical to its `CV Macro Precision SD`.

They should be confirmed with the responsible teammate before using a CV-stability ranking in the final presentation.
