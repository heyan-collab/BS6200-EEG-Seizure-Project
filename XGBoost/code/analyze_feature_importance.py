"""Extract gain importance from the saved booster without training or inference."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=root / 'results/final/model.json')
    parser.add_argument('--output-dir', type=Path, default=root / 'results/interpretability')
    parser.add_argument('--figure-dir', type=Path, default=root / 'results/figures')
    parser.add_argument('--report', type=Path, default=root / 'docs/FEATURE_IMPORTANCE.md')
    args = parser.parse_args()
    names = ['feature_importance_top20_gain.png', 'feature_importance_gain_by_position.png']
    if (args.output_dir.exists() and any(args.output_dir.iterdir())) or args.report.exists() or any((args.figure_dir / n).exists() for n in names):
        raise FileExistsError('Existing interpretation outputs would be overwritten; use new paths')
    original_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    model = xgb.Booster()
    model.load_model(args.model)
    expected = [f'x{i}' for i in range(1, 179)]
    if model.num_features() != 178 or model.feature_names != expected:
        raise ValueError('Saved model feature names must match x1–x178 exactly in order')
    # Native gain is mean split gain per feature, not total_gain or split count.
    score = model.get_score(importance_type='gain')
    if not set(score).issubset(expected):
        raise ValueError('Unexpected importance feature keys')
    positions = pd.DataFrame({'feature': expected, 'position': range(1, 179),
                              'gain': [float(score.get(f, 0.0)) for f in expected]})
    if not np.isfinite(positions.gain).all() or (positions.gain < 0).any():
        raise ValueError('Invalid gain values')
    ranked = positions.sort_values(['gain', 'position'], ascending=[False, True]).reset_index(drop=True)
    ranked.insert(0, 'rank', range(1, 179))
    top = ranked.head(20)
    regions = []
    for start, end in [(1, 59), (60, 118), (119, 178)]:
        block = positions[positions.position.between(start, end)]
        regions.append({'position_start': start, 'position_end': end,
                        'n_features': len(block), 'mean_feature_gain': float(block.gain.mean()),
                        'top20_features': int(top.position.between(start, end).sum())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output_dir / 'feature_importance_gain_ranked.csv', index=False)
    summary = {'importance_type': 'gain', 'definition': 'Average split loss-reduction gain for each feature across the fitted ensemble',
               'model_sha256': original_hash, 'feature_names_verified': True, 'n_features': 178,
               'n_zero_gain_features': int((positions.gain == 0).sum()),
               'tie_break': 'Ascending feature position', 'gain_normalized': False,
               'regions': regions, 'top20': top.to_dict(orient='records'),
               'xgboost_version': xgb.__version__, 'matplotlib_version': matplotlib.__version__,
               'training_performed': False, 'inference_performed': False, 'test_labels_used': False}
    (args.output_dir / 'feature_importance_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 11, 'axes.titlepad': 14})
    fig, ax = plt.subplots(figsize=(8.5, 7.5), layout='constrained')
    bars = ax.barh(top.feature, top.gain, color='#0072B2', height=0.72)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f'{v:.3f}' for v in top.gain], padding=4, fontsize=9)
    ax.set(xlabel='Gain importance (average split gain)', ylabel='EEG feature',
           title='Final XGBoost model: top 20 features by gain', xlim=(0, top.gain.max() * 1.18))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#e2e6e9', linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(args.figure_dir / names[0], dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.8), layout='constrained')
    ax.plot(positions.position, positions.gain, color='#0072B2', linewidth=1.2,
            marker='o', markersize=2.5, label='All 178 positions')
    ax.scatter(top.position, top.gain, color='#D55E00', marker='D', s=25, zorder=3, label='Top 20 features')
    ticks = [1, 20, 40, 60, 80, 100, 120, 140, 160, 178]
    ax.set(xticks=ticks, xticklabels=[f'x{i}' for i in ticks], xlim=(0, 179), ylim=(0, positions.gain.max() * 1.15),
           xlabel='EEG measurement position', ylabel='Gain importance (average split gain)',
           title='Final XGBoost model: gain across the EEG segment')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e2e6e9', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper right')
    fig.savefig(args.figure_dir / names[1], dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    lines = ['# Final model feature importance', '',
             'The saved final model was loaded from `results/final/model.json`.',
             'Before importance extraction, its feature names were verified to be',
             '**exactly `x1`–`x178` in order**, with 178 input features.',
             'No training, tuning, model inference, dataset reads or test-label access',
             'was performed. Existing experiment outputs were not modified.', '',
             '## Method', '',
             "Importance was extracted with `Booster.get_score(importance_type='gain')`.",
             'Gain is the average loss-reduction gain of the splits that use a feature,',
             'aggregated across the fitted multiclass ensemble. It is not the number',
             'of splits (`weight`) or their summed gains (`total_gain`). Raw gain values',
             'are reported without normalization. Features omitted by XGBoost because',
             'they are unused receive zero gain. Ranking is descending gain, with ties',
             'resolved by ascending measurement position.', '',
             f"All 178 features are included; {summary['n_zero_gain_features']} have zero gain.", '',
             '## Top 20 features', '', '| Rank | Feature | Position | Gain |', '|---:|---|---:|---:|']
    for row in top.itertuples(index=False):
        lines.append(f'| {row.rank} | `{row.feature}` | {row.position} | {row.gain:.6f} |')
    lines += ['', '![Top 20 features by gain](../results/figures/' + names[0] + ')', '',
              'Bars are ordered from highest gain at the top to lowest among the top 20.', '',
              '## Importance across measurement positions', '',
              '![Gain across EEG positions](../results/figures/' + names[1] + ')', '',
              'The x-axis follows the original within-window measurement order, rather',
              'than importance rank. Orange diamonds identify the top 20 features.',
              'Connecting lines help trace positions; they do not interpolate EEG signals.', '',
              'A descriptive breakdown into three approximately equal position ranges:', '',
              '| Positions | Features | Mean feature gain | Top-20 count |', '|---|---:|---:|---:|']
    for r in regions:
        lines.append(f"| x{r['position_start']}–x{r['position_end']} | {r['n_features']} | {r['mean_feature_gain']:.6f} | {r['top20_features']} |")
    first = top.iloc[0]
    lines += ['', f"The highest-gain feature is **{first.feature}** (gain **{first.gain:.6f}**).",
              'The range counts and means describe this fitted model only. They are',
              'not a statistical test of temporal concentration or a clinical finding.', '',
              '## Interpretation limits', '',
              '**XGBoost gain importance reflects the fitted model’s use of features;',
              'it does not establish causal or clinical importance.** These features',
              'are measurement positions within a window, not separate anatomical',
              'regions or named clinical biomarkers. Correlated neighboring positions',
              'can substitute for each other, so gain can be distributed unevenly.',
              'Average gain also does not directly quantify a feature’s effect on',
              'accuracy, feature-use frequency, effect direction, or class-specific',
              'contribution. This analysis supplies no uncertainty intervals.', '',
              'The existing split has inferred recording-group overlap. Interpretation',
              'does not resolve that limitation or demonstrate generalization to unseen',
              'recordings or subjects. No feature selection was performed.', '',
              '## Files and reproduction', '',
              '- `src/analyze_feature_importance.py`: verifies feature mapping, extracts',
              '  native gain scores, ranks every feature, creates figures and this report.',
              '- `results/interpretability/feature_importance_gain_ranked.csv`: all 178',
              '  ranks, feature names, positions and raw gains.',
              '- `results/interpretability/feature_importance_summary.json`: top 20,',
              '  position-range summaries, model checksum and analysis metadata.',
              '- Both figures are 300 dpi PNGs under `results/figures/`.', '',
              'With XGBoost, NumPy, pandas and Matplotlib installed, run from the root:', '',
              '```sh', 'python src/analyze_feature_importance.py', '```', '',
              'In the current cloud instance, prefix with',
              '`PYTHONPATH=/workspace/scratch/xgboost-baseline-deps`.',
              'The script refuses existing interpretation outputs. For regeneration,',
              'provide new `--output-dir`, `--figure-dir` and `--report` paths.', '',
              f'Model SHA-256: `{original_hash}`.', '']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines))
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError('Model bytes changed during analysis')
    print(top.to_string(index=False))
    print('Position ranges:', regions)
    print('Model feature mapping verified; no training, inference or label access.')


if __name__ == '__main__':
    main()
