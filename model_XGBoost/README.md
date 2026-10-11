# BS6200-project

Five-class EEG classification with a fixed shared train/test split and an
XGBoost baseline.

```text
data/                 Dataset and shared train/test and CV splits
src/                  xgboost_baseline.py
results/baseline/     Saved model, predictions, and metrics
docs/                 Baseline methodology and performance report
requirements.txt      Pinned Python dependencies
```

See [baseline instructions](docs/BASELINE.md) for setup, code explanation,
and training commands, and [baseline results](docs/BASELINE_RESULTS.md)
for the existing performance report and potential same-source leakage caveat.

The saved artifacts are in [results/baseline](results/baseline/).
Loading the existing model does not require retraining:

```python
import xgboost as xgb

model = xgb.Booster()
model.load_model("results/baseline/model.json")
```

Run commands from the repository root. The baseline script defaults to
`data/` for inputs and `results/baseline/` for outputs. Executing
`python src/xgboost_baseline.py` trains and writes artifacts; it is unnecessary
when inspecting the saved baseline.
