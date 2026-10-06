# CNN-Transformer Summary

## Model and Training
- Input: raw single-channel EEG, X1-X178.
- CNN: two blocks with 32/64 channels and kernel sizes 7/5, batch normalization, GELU, and max pooling.
- Transformer: learnable positional embeddings, 2 encoder layers, 4 attention heads, embedding dimension 64, feed-forward dimension 128.
- Classifier: concatenated temporal mean/max pooling, dropout, and a linear layer.
- Adam, learning rate 0.001, batch size 64, cross-entropy loss, CPU training.
- Shared 80/20 development/test split and 5 development-set CV folds; seed 6200.
- StandardScaler fitted only on the training partition of each fold; final scaler fitted on development data.
- Final training epochs: binary 22; five-class 32, selected as the median best epoch across CV folds.

## Results
CV values are mean +/- standard deviation across five folds.

| Task | CV primary F1 | Final test primary F1 | Final test accuracy | Final test ROC-AUC |
|---|---:|---:|---:|---:|
| Binary (seizure = 1) | 0.9859 +/- 0.0021 | 0.9751 | 0.9900 | 0.9991 |
| Five-class | 0.8135 +/- 0.0094 | 0.7805 | 0.7878 | 0.9621 |

Binary F1 refers to the seizure class; five-class F1 is Macro F1, and five-class ROC-AUC is macro one-vs-rest.
Binary test errors: 10 missed seizures and 13 false alarms.
Five-class test errors: 488 total, including 239 confusions between classes 2 and 3.

## Interpretation and Limitations
- Occlusion sensitivity identifies input windows influencing predictions; it does not establish physiological causality.
- Some incorrect predictions have high confidence; probability calibration was not evaluated.
- Reliable recording/patient IDs were unavailable, so sample-level splitting cannot rule out related-recording leakage or establish unseen-patient generalization.
- Shared row indices require the original dataset row order to remain unchanged.
- The test set was inspected during project development; this limits claims of a completely untouched final evaluation. Further selection should use development data only.
- CV prediction checkpoints were selected using their validation folds; ensemble evaluation should account for this selection and use development data to choose weights.

## Files
- train_cnn_transformer.py
- final_train_test_transformer.py
- analyze_cnn_transformer.py
- outputs/cnn_transformer/: CV metrics, predictions, configuration, and checkpoints.
- outputs/cnn_transformer_final/: final metrics, predictions, confusion matrices, and interpretation outputs.

## References
- Vaswani et al. (2017), Attention Is All You Need: https://arxiv.org/abs/1706.03762
- UCI Epileptic Seizure Recognition dataset: https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition
