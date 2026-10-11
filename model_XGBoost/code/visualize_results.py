"""Create figures and summary tables from saved experiment results only.

Dependencies: numpy, pandas, matplotlib. No datasets or model files are loaded.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

COLORS = ['#0072B2', '#E69F00', '#009E73']  # Colorblind-friendly palette.
DPI = 300


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e2e6e9', linewidth=0.6)
    ax.set_axisbelow(True)


def create_figures(out, final, baseline, folds, best):
    # 1. Count heatmap: preserve true/predicted orientation and annotate zeros too.
    matrix = np.array(final['test']['confusion_matrix'])
    if matrix.shape != (5, 5) or matrix.sum() != 2300:
        raise ValueError('Unexpected saved test confusion matrix')
    fig, ax = plt.subplots(figsize=(6.6, 5.6), layout='constrained')
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=matrix.max())
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max() * 0.55 else '#172b3a', fontsize=12)
    ax.set(xticks=range(5), yticks=range(5), xticklabels=range(1, 6), yticklabels=range(1, 6),
           xlabel='Predicted class', ylabel='True class', title='Final test confusion matrix')
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    fig.colorbar(im, ax=ax, shrink=0.85, label='Sample count')
    save(fig, out / 'final_test_confusion_matrix.png')

    # 2. Per-class metrics directly from the saved classification report.
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    x = np.arange(5)
    width = 0.24
    for offset, (key, label, color) in enumerate(zip(['precision', 'recall', 'f1-score'], ['Precision', 'Recall', 'F1-score'], COLORS)):
        values = [final['test']['classification_report'][str(k)][key] for k in range(1, 6)]
        bars = ax.bar(x + (offset - 1) * width, values, width, label=label, color=color)
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=[f'Class {k}' for k in range(1, 6)], ylim=(0, 1),
           ylabel='Score', title='Final test performance by class')
    style_axis(ax)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    save(fig, out / 'final_test_per_class_performance.png')

    # 3. Keep a full 0–100% scale; compare the same saved test metrics.
    names = ['Accuracy', 'Macro\nprecision', 'Macro\nrecall', 'Macro F1']
    keys = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    x = np.arange(4)
    for offset, (r, label, color) in enumerate([(baseline, 'Baseline', COLORS[0]), (final, 'Tuned final', COLORS[1])]):
        values = [r['test'][k] for k in keys]
        bars = ax.bar(x + (offset - 0.5) * 0.34, values, 0.34, label=label, color=color)
        ax.bar_label(bars, labels=[f'{v:.2%}' for v in values], padding=4, fontsize=9)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel='Test performance', title='Baseline and tuned final model')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    style_axis(ax)
    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.18))
    save(fig, out / 'baseline_vs_final_test_performance.png')

    # 4. Two panels avoid occlusion of the almost identical F1/accuracy values.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout='constrained', sharey=True)
    for ax, metric, label, color, marker in zip(axes, ['macro_f1', 'accuracy'], ['Macro F1', 'Accuracy'], COLORS[:2], ['o', 's']):
        values = folds[metric].to_numpy()
        mean = values.mean()
        ax.plot(folds['fold'], values, marker=marker, color=color, linewidth=1.8, markersize=6, label='Validation fold')
        ax.axhline(mean, color='#535c64', linestyle='--', linewidth=1.3, label=f'Mean = {mean:.4f}')
        for fold, value in zip(folds['fold'], values):
            ax.annotate(f'{value:.4f}', (fold, value), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
        ax.set(xticks=range(5), xlabel='Shared CV fold', title=label, ylim=(0, 1), xlim=(-0.35, 4.35))
        style_axis(ax)
        ax.legend(frameon=False, loc='lower center')
    axes[0].set_ylabel('Validation score')
    fig.suptitle(f"Selected configuration: five-fold CV (candidate {best['candidate_id']})")
    save(fig, out / 'selected_five_fold_cv_performance.png')


def create_summary(path, final, baseline, best, folds, out):
    lines = ['# XGBoost results summary', '',
             'All figures and tables below use saved experiment results. No training,',
             'tuning, model inference, split modification or result recomputation from',
             'EEG data was performed. Figures are exported as 300 dpi PNG files.', '',
             '## Selected XGBoost hyperparameters', '',
             'Candidate 7 was selected using mean five-fold CV macro F1 only.', '',
             '| Parameter | Saved value |', '|---|---|']
    for key, value in {**best['parameters'], **best['fixed_parameters']}.items():
        lines.append(f'| `{key}` | `{value}` |')
    lines += ['', '## Final overall test performance', '',
              'Test support: **2,300 samples**, using the unchanged shared split.', '',
              '| Metric | Baseline test | Final test | Difference (percentage points) |', '|---|---:|---:|---:|']
    for key, label in [('accuracy', 'Accuracy'), ('macro_precision', 'Macro precision'), ('macro_recall', 'Macro recall'), ('macro_f1', 'Macro F1')]:
        a, b = baseline['test'][key], final['test'][key]
        lines.append(f'| {label} | {a:.4%} | {b:.4%} | {100 * (b-a):+.4f} |')
    lines += ['', '## Final per-class test performance', '',
              '| Class | Precision | Recall | F1-score | Support |', '|---|---:|---:|---:|---:|']
    for k in range(1, 6):
        r = final['test']['classification_report'][str(k)]
        lines.append(f"| {k} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |")
    figures = [
        ('final_test_confusion_matrix.png', 'Final test confusion matrix', 'Counts show true classes on rows and predictions on columns. Classes 2 and 3 are frequently confused (165 class-2 samples predicted as 3 and 114 class-3 samples predicted as 2); 109 class-4 samples were predicted as class 5.'),
        ('final_test_per_class_performance.png', 'Per-class final test performance', 'Precision, recall and F1 use a 0–1 scale. Class 1 has the highest F1 (0.9365), and class 2 has the lowest (0.5552). Each class has 460 test samples.'),
        ('baseline_vs_final_test_performance.png', 'Baseline versus tuned final model', 'The same saved test metrics are compared on a full 0–100% scale. Final accuracy is higher by 0.7391 percentage points and macro F1 by 0.6244 percentage points. These are descriptive differences, without a significance claim.'),
        ('selected_five_fold_cv_performance.png', 'Selected configuration: five-fold CV', f"Each panel shows one metric for folds 0–4; dashed lines show arithmetic means. Macro F1 is {best['mean_cv_macro_f1']:.4f} (SD {best['std_cv_macro_f1']:.4f}); accuracy is {best['mean_cv_accuracy']:.4f} (SD {best['std_cv_accuracy']:.4f}). SD uses ddof=0. The configuration was selected among 12 candidates, so these CV results can have selection optimism."),
    ]
    import os
    for filename, heading, caption in figures:
        relative = Path(os.path.relpath(out / filename, path.parent)).as_posix()
        lines += ['', f'## {heading}', '', f'![{heading}]({relative})', '', caption]
    lines += ['', '## Interpretation and reproduction', '',
              'All training metrics for the final model are 1.0, while test accuracy',
              'is 0.7013. The saved shared split has inferred recording-group overlap:',
              'every test window has same-group windows in development, and every CV',
              'validation window has same-group windows in its training portion.',
              'These scores do not establish generalization to unseen recordings or',
              'subjects. Classes remain numbered 1–5; no undocumented clinical mapping',
              'or statistical significance is claimed.', '',
              'See [final results](FINAL_RESULTS.md), [tuning report](TUNING.md),',
              'and [baseline results](BASELINE_RESULTS.md) for the underlying experiment.', '',
              'The plotting script reads only `results/baseline/metrics.json`,',
              '`results/final/metrics.json`, `results/tuning/best_parameters.json`,',
              'and `results/tuning/fold_scores.csv`. It imports no training code.', '',
              'From the repository root, with NumPy, pandas and Matplotlib installed:', '',
              '```sh', 'python src/visualize_results.py', '```', '',
              'The script refuses nonempty figure directories or an existing summary.',
              'For regeneration, supply a new empty `--output-dir` and a new `--summary`',
              'path. Existing experiment outputs are never written.', '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=root / 'results')
    parser.add_argument('--output-dir', type=Path, default=root / 'results/figures')
    parser.add_argument('--summary', type=Path, default=root / 'docs/RESULTS_SUMMARY.md')
    args = parser.parse_args()
    if (args.output_dir.exists() and any(args.output_dir.iterdir())) or args.summary.exists():
        raise FileExistsError('Use empty figure directory and new summary path; existing files are preserved')
    final = json.loads((args.results_dir / 'final/metrics.json').read_text())
    baseline = json.loads((args.results_dir / 'baseline/metrics.json').read_text())
    best = json.loads((args.results_dir / 'tuning/best_parameters.json').read_text())
    folds = pd.read_csv(args.results_dir / 'tuning/fold_scores.csv')
    folds = folds[folds.candidate_id == best['candidate_id']].sort_values('fold')
    if len(folds) != 5 or list(folds.fold) != list(range(5)):
        raise ValueError('Expected exactly five saved folds for the selected candidate')
    for key in ['macro_f1', 'accuracy']:
        if not np.isclose(folds[key].mean(), best[f'mean_cv_{key}']):
            raise ValueError('Selected fold scores disagree with saved mean')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 11,
                         'axes.titlepad': 16, 'axes.linewidth': 0.7,
                         'savefig.dpi': DPI, 'figure.dpi': 120})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    create_figures(args.output_dir, final, baseline, folds, best)
    create_summary(args.summary, final, baseline, best, folds, args.output_dir)
    print('Saved four 300 dpi figures and summary tables; only saved metric files were read.')


if __name__ == '__main__':
    main()
