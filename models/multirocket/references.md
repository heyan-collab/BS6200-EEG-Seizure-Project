# References

## MultiRocket

Tan, C. W., Dempster, A., Bergmeir, C., and Webb, G. I. (2022). MultiRocket: multiple pooling operators and transformations for fast and effective time series classification. *Data Mining and Knowledge Discovery*, 36, 1623–1646.

- DOI: https://doi.org/10.1007/s10618-022-00844-1
- Author repository: https://github.com/ChangWeiTan/MultiRocket

## Software implementation

aeon developers. `MultiRocketClassifier`, aeon 1.6.0.

- Repository: https://github.com/aeon-toolkit/aeon
- Documentation: https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.convolution_based.MultiRocketClassifier.html
- License: BSD-3-Clause

## Project implementation note

This project uses aeon's open-source `MultiRocketClassifier`. The data loading, fixed-fold evaluation, metrics, result export, and figures were written for the BS6200 project. The original EEG values are reshaped from `(n_samples, 178)` to `(n_samples, 1, 178)` without changing their temporal order.
