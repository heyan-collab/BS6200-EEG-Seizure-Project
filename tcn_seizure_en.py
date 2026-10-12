import os
import glob
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, precision_recall_fscore_support,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


FINAL_ONLY = False
FINAL_CHANNELS = 32
FINAL_CLASS_WEIGHT = False
FINAL_MULTI_CHANNELS = 32

EVALUATE_FINAL_TEST = False
QUICK_MODE = False

SEED = 6200
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 1e-3
EARLY_STOP_FRACTION = 0.1
VERBOSE = False


DEFAULT_CHANNELS = 32
NUM_LEVELS = 5
KERNEL_SIZE = 5
DROPOUT = 0.2

CHANNEL_GRID = [16, 32, 64]
THRESHOLD = 0.5

OUTPUT_DIR = "results"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


CLASS_NAMES = {
    1: "Seizure activity",
    2: "Tumour region",
    3: "Healthy brain region of a patient with a tumour",
    4: "Eyes closed",
    5: "Eyes open",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_file(name):
    for folder in ["/kaggle/input", "/content", "."]:
        hits = glob.glob(f"{folder}/**/{name}", recursive=True)
        if hits:
            print("Found file:", hits[0])
            return hits[0]
    raise FileNotFoundError(f"Could not find {name}. Please place it next to this script or add it to the notebook inputs.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    df = pd.read_csv(find_file("Epileptic Seizure Recognition.csv"))
    X = df[[f"X{i}" for i in range(1, 179)]].values.astype(np.float32)
    y_raw = df["y"].values.astype(int)

    split_df = pd.read_csv(find_file("shared_split_with_cv.csv")).set_index("row_index")
    split_df = split_df.loc[np.arange(len(df))]
    split = split_df["split"].values
    fold = split_df["cv_fold"].fillna(-1).astype(int).values
    return X, y_raw, split, fold


def make_labels(y_raw, task):
    return (y_raw == 1).astype(int) if task == "binary" else y_raw - 1


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + res)


class TCN(nn.Module):
    def __init__(self, n_outputs, channels, levels=NUM_LEVELS,
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        blocks = []
        for i in range(levels):
            in_ch = 1 if i == 0 else channels
            blocks.append(TemporalBlock(in_ch, channels, kernel_size, 2 ** i, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(channels, n_outputs)

    def forward(self, x):
        h = self.tcn(x)[:, :, -1]
        return self.fc(h)


def receptive_field(levels=NUM_LEVELS, kernel_size=KERNEL_SIZE):
    return 1 + 2 * (kernel_size - 1) * (2 ** levels - 1)


def make_loader(X, y, shuffle):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE,
                      shuffle=shuffle, drop_last=shuffle)


def make_criterion(task, y_train, class_weight):
    if task == "binary":
        pw = None
        if class_weight:
            pw = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()],
                              dtype=torch.float32, device=DEVICE)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return nn.CrossEntropyLoss()


def compute_loss(criterion, logits, yb, task):
    if task == "binary":
        return criterion(logits.squeeze(1), yb.float())
    return criterion(logits, yb.long())


def train_one_model(task, cfg, X_tr, y_tr, X_es, y_es):
    n_out = 1 if task == "binary" else 5
    model = TCN(n_out, cfg["channels"]).to(DEVICE)
    criterion = make_criterion(task, y_tr, cfg["class_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(X_tr, y_tr, shuffle=True)
    es_loader = make_loader(X_es, y_es, shuffle=False)

    history = {"train_loss": [], "es_loss": [], "best_epoch": 0}
    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = compute_loss(criterion, model(xb), yb, task)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / (len(train_loader) * BATCH_SIZE)


        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in es_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                total += compute_loss(criterion, model(xb), yb, task).item() * len(xb)
        es_loss = total / len(es_loader.dataset)

        history["train_loss"].append(train_loss)
        history["es_loss"].append(es_loss)
        if VERBOSE:
            print(f"      epoch {epoch:3d} | train {train_loss:.4f} | early-stop {es_loss:.4f}")


        if es_loss < best_loss:
            best_loss, bad_epochs = es_loss, 0
            history["best_epoch"] = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_proba(model, X, task):
    model.eval()
    out = []
    for xb, _ in make_loader(X, np.zeros(len(X), dtype=np.int64), shuffle=False):
        logits = model(xb.to(DEVICE))
        p = torch.sigmoid(logits.squeeze(1)) if task == "binary" else torch.softmax(logits, 1)
        out.append(p.cpu().numpy())
    return np.concatenate(out)


BINARY_METRICS = ["Accuracy", "Balanced accuracy", "Precision", "Recall",
                  "Specificity", "F1", "AUROC", "Average precision"]
MULTI_METRICS = ["Accuracy", "Balanced accuracy", "Macro precision", "Macro recall",
                 "Macro F1", "Macro ROC-AUC (one-vs-rest)"]


def binary_metrics(y_true, y_prob):
    y_pred = (y_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y_true, y_prob),
        "Average precision": average_precision_score(y_true, y_prob),
    }


def multi_metrics(y_true, prob):
    y_pred = prob.argmax(1)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro ROC-AUC (one-vs-rest)": roc_auc_score(y_true, prob, multi_class="ovr",
                                                     average="macro"),
    }


def run_cv(task, cfg, X, y_raw, split, fold):
    set_seed(SEED)
    y = make_labels(y_raw, task)
    dev_idx = np.where(split == "development")[0]
    test_idx = np.where(split == "final_test")[0]
    n_cls = 1 if task == "binary" else 5
    oof = np.zeros(len(dev_idx)) if task == "binary" else np.zeros((len(dev_idx), n_cls))
    test_probs, rows, histories = [], [], []
    metric_fn = binary_metrics if task == "binary" else multi_metrics
    t0 = time.time()

    print(f"\n>>> [{task}] {cfg['name']}")
    for k in range(5):
        fold_t0 = time.time()
        tr_all = dev_idx[fold[dev_idx] != k]
        va = dev_idx[fold[dev_idx] == k]

        tr, es = train_test_split(tr_all, test_size=EARLY_STOP_FRACTION,
                                  stratify=y[tr_all], random_state=SEED + k)


        mean, std = X[tr].mean(), X[tr].std()
        norm = lambda a: (a - mean) / std

        model, hist = train_one_model(task, cfg, norm(X[tr]), y[tr], norm(X[es]), y[es])
        histories.append(hist)

        p = predict_proba(model, norm(X[va]), task)
        oof[fold[dev_idx] == k] = p
        m = metric_fn(y[va], p)
        rows.append({"fold": k, **m})
        key = "F1" if task == "binary" else "Macro F1"
        print(f"    fold {k}: {key} = {m[key]:.4f}  (best epoch {hist['best_epoch']})")

        if EVALUATE_FINAL_TEST:
            test_probs.append(predict_proba(model, norm(X[test_idx]), task))
        hist["seconds"] = time.time() - fold_t0

    fold_df = pd.DataFrame(rows).set_index("fold")
    summary = pd.DataFrame({"Mean": fold_df.mean(), "Standard deviation": fold_df.std()})
    seconds = time.time() - t0
    fold_df.to_csv(os.path.join(OUTPUT_DIR, f"cv_{task}_{cfg['name']}.csv"))
    print(f"    Mean {key} = {summary.loc[key, 'Mean']:.4f} ± "
          f"{summary.loc[key, 'Standard deviation']:.4f}  (time: {seconds/60:.1f} min)")

    return {
        "task": task, "cfg": cfg, "fold_df": fold_df, "summary": summary,
        "oof_prob": oof, "oof_true": y[dev_idx], "histories": histories,
        "test_prob": np.mean(test_probs, axis=0) if test_probs else None,
        "test_true": y[test_idx], "seconds": seconds,
        "dev_idx": dev_idx, "test_idx": test_idx,
    }


def make_cfg(channels, class_weight):
    cw = "balanced" if class_weight else "noweight"
    return {"channels": channels, "class_weight": class_weight, "name": f"ch{channels}_{cw}"}


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.frameon": False,
    "lines.linewidth": 2, "font.size": 9.5,
})


def savefig(fig, name, folder=None):
    path = os.path.join(folder or FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return f"figures/{name}"


def plot_example_signals(X, y_raw, split):
    fig, axes = plt.subplots(5, 1, figsize=(8, 7.5), sharex=True, sharey=True)
    for ax, c in zip(axes, range(1, 6)):
        i = np.where((y_raw == c) & (split == "development"))[0][0]
        ax.plot(np.arange(1, 179), X[i], color=SERIES[0], linewidth=1.2)
        ax.set_title(f"Class {c}: {CLASS_NAMES[c]}", loc="left", fontsize=10)
        ax.set_ylabel("µV")
    axes[-1].set_xlabel("Sample index (178 samples ≈ 1 s)")
    return savefig(fig, "example_signals.png")


def plot_binary_curves(res):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    fold_ids = res["fold_df"].index
    dev_fold = FOLD_OF_DEV
    for k in fold_ids:
        m = dev_fold == k
        yt, yp = res["oof_true"][m], res["oof_prob"][m]
        fpr, tpr, _ = roc_curve(yt, yp)
        pr, rc, _ = precision_recall_curve(yt, yp)
        ax[0].plot(fpr, tpr, color=SERIES[k], label=f"Fold {k} (AUROC {roc_auc_score(yt, yp):.4f})")
        ax[1].plot(rc, pr, color=SERIES[k], label=f"Fold {k} (AP {average_precision_score(yt, yp):.4f})")
    ax[0].plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (validation folds)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves (validation folds)")
    for a in ax:
        a.legend(loc="lower right" if a is ax[0] else "lower left", fontsize=8.5)
    return savefig(fig, "binary_roc_pr_cv.png")


def plot_confusion(cm, labels, title, name, figsize=(4.6, 4), folder=None):
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}", ha="center", va="center",
                    fontsize=8.5, color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    for s in ax.spines.values():
        s.set_visible(False)
    return savefig(fig, name, folder)


def plot_fold_metrics(res, metrics, name, title):
    df = res["fold_df"][metrics]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for j, m in enumerate(metrics):
        for k in df.index:
            ax.scatter(j + (k - 2) * 0.07, df.loc[k, m], s=36, color=SERIES[k],
                       edgecolor=SURFACE, linewidth=1, zorder=3,
                       label=f"Fold {k}" if j == 0 else None)
        ax.plot([j - 0.25, j + 0.25], [df[m].mean()] * 2, color=INK, linewidth=1.5)
    ax.set_xticks(range(len(metrics)), [m.replace(" (one-vs-rest)", "") for m in metrics],
                  rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title + "  (dots = folds, bar = mean)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    return savefig(fig, name)


def plot_training_curves(res, name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for k, h in enumerate(res["histories"]):
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax[0].plot(ep, h["train_loss"], color=SERIES[k], label=f"Fold {k}")
        ax[1].plot(ep, h["es_loss"], color=SERIES[k])
        b = h["best_epoch"]
        ax[1].scatter(b, h["es_loss"][b - 1], s=40, color=SERIES[k], edgecolor=SURFACE, zorder=3)
    ax[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    ax[1].set(title="Early-stopping loss (dot = selected epoch)", xlabel="Epoch")
    ax[1].legend(*ax[0].get_legend_handles_labels(), fontsize=8.5, loc="upper right")
    return savefig(fig, name)


def plot_config_bars(results, labels, metrics, name, title):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    n = len(results)
    width = 0.8 / n
    for i, (r, lab) in enumerate(zip(results, labels)):
        means = [r["summary"].loc[m, "Mean"] for m in metrics]
        sds = [r["summary"].loc[m, "Standard deviation"] for m in metrics]
        x = np.arange(len(metrics)) + (i - (n - 1) / 2) * width
        ax.bar(x, means, width * 0.92, yerr=sds, color=SERIES[i], label=lab,
               error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    lo = min(r["summary"].loc[m, "Mean"] for r in results for m in metrics)
    ax.set_ylim(max(0, lo - 0.05), 1.0)
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylabel("Mean over 5 folds")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    return savefig(fig, name)


def plot_channel_study(bin_runs, multi_runs, name):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, runs, key, ttl in [(ax[0], bin_runs, "F1", "Binary: F1"),
                              (ax[1], multi_runs, "Macro F1", "Five-class: Macro F1")]:
        chs = sorted(runs)
        mean = [runs[c]["summary"].loc[key, "Mean"] for c in chs]
        sd = [runs[c]["summary"].loc[key, "Standard deviation"] for c in chs]
        a.errorbar(chs, mean, yerr=sd, color=SERIES[0], marker="o", markersize=7,
                   capsize=3, ecolor=INK2, elinewidth=1)
        for c, m in zip(chs, mean):
            a.annotate(f"{m:.4f}", (c, m), textcoords="offset points", xytext=(8, 0),
                       ha="left", va="center", fontsize=8.5, color=INK2)
        a.set_xscale("log", base=2)
        a.set_xticks(chs, [str(c) for c in chs])
        a.set_xlim(min(chs) / 1.5, max(chs) * 1.8)
        a.set(title=ttl + " vs channels per layer", xlabel="Channels per layer",
              ylabel="Mean ± SD over 5 folds")
    return savefig(fig, name)


def f4(x):
    return f"{x:.4f}"


def md_table(header, rows, align, bold_cols=None, bold_min_cols=None):
    rows = [[str(v) for v in r] for r in rows]
    if len(rows) > 1:
        for cols, pick in [(bold_cols or [], max), (bold_min_cols or [], min)]:
            for c in cols:
                target = pick(float(r[c]) for r in rows)
                for r in rows:
                    if float(r[c]) == target:
                        r[c] = f"**{r[c]}**"
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def summary_table(res, metrics, bold_metric):
    rows = []
    for m in metrics:
        mean, sd = f4(res["summary"].loc[m, "Mean"]), f4(res["summary"].loc[m, "Standard deviation"])
        if m == bold_metric:
            mean, sd = f"**{mean}**", f"**{sd}**"
        rows.append([m, mean, sd])
    return md_table(["Metric", "Mean", "Standard deviation"], rows, ["l", "r", "r"])


def write_report(ctx):
    fb, fm = ctx["final_bin"], ctx["final_multi"]
    cfg = fb["cfg"]
    L = []
    L.append("# TCN experiment results\n")


    L.append("## Selected model\n")
    L += [
        "- Task: binary seizure recognition (`y == 1` versus `y in {2, 3, 4, 5}`)",
        "- Input: raw ordered EEG values `X1` to `X178`, standardised with the mean and SD of each training fold",
        "- Validation: the five fixed development folds in `shared_split_with_cv.csv`",
        f"- Model: temporal convolutional network (PyTorch), {NUM_LEVELS} residual blocks, "
        f"dilations 1–{2 ** (NUM_LEVELS - 1)}, kernel size {KERNEL_SIZE}, dropout {DROPOUT}",
        f"- Channels per layer: {cfg['channels']}",
        f"- Receptive field: {receptive_field()} samples (covers the full 178-sample segment)",
        f"- Class weight: {'balanced (pos_weight = negatives / positives)' if cfg['class_weight'] else 'none'}",
        f"- Optimiser: Adam, learning rate {LEARNING_RATE}, batch size {BATCH_SIZE}, "
        f"up to {MAX_EPOCHS} epochs, early stopping patience {PATIENCE}",
        f"- Early stopping: on a stratified {int(EARLY_STOP_FRACTION * 100)}% split of the training folds "
        "(the validation fold is never used for training or model selection within a fold)",
        f"- Random seed: {SEED}",
        "- Primary selection metric: seizure-class F1",
        f"- Final test accessed: {'yes' if EVALUATE_FINAL_TEST else 'no'}",
        "",
    ]


    L.append("## Final five-fold cross-validation\n")
    L.append(summary_table(fb, BINARY_METRICS, "F1") + "\n")
    f1s = ", ".join(f4(v) for v in fb["fold_df"]["F1"])
    sd = fb["summary"].loc["F1", "Standard deviation"]
    L.append(f"The fold-level F1 scores were {f1s}. "
             f"The standard deviation across folds was {f4(sd)}.\n")
    L.append(f"![ROC and PR curves]({ctx['fig_roc']})\n")
    L.append(f"![Fold-level metrics]({ctx['fig_bin_folds']})\n")
    L.append(f"![Binary confusion matrix]({ctx['fig_bin_cm']})\n")
    cm = ctx["bin_cm"]
    L.append(f"Pooled over the five validation folds, {cm[1, 1]} of {cm[1].sum()} seizure segments "
             f"were detected ({cm[1, 0]} missed), and {cm[0, 1]} of {cm[0].sum()} non-seizure "
             f"segments were falsely flagged as seizures.\n")
    L.append(f"![Training curves]({ctx['fig_train']})\n")
    epochs = [h["best_epoch"] for h in fb["histories"]]
    L.append(f"Selected epochs per fold: {', '.join(map(str, epochs))}. "
             f"Training the five folds took {fb['seconds'] / 60:.1f} minutes on `{DEVICE}`.\n")


    abl = ctx["ablation"]
    if len(abl) > 1:
        L.append(f"## Class-weight ablation ({abl[False]['cfg']['channels']} channels)\n")
        rows = []
        for cw, lab in [(False, "No class weight"), (True, "Balanced class weight")]:
            s = abl[cw]["summary"]["Mean"]
            rows.append([lab] + [f4(s[m]) for m in ["F1", "Precision", "Recall", "Specificity"]])
        L.append(md_table(["Configuration", "F1", "Precision", "Recall", "Specificity"],
                          rows, ["l", "r", "r", "r", "r"], bold_cols=[1, 2, 3, 4]) + "\n")
        s0, s1 = abl[False]["summary"]["Mean"], abl[True]["summary"]["Mean"]
        chosen = "balanced" if ctx["best_cw"] else "no"
        L.append(f"Relative to no weighting, balanced weighting changed recall by "
                 f"{s1['Recall'] - s0['Recall']:+.4f} and precision by "
                 f"{s1['Precision'] - s0['Precision']:+.4f}, giving F1 = {f4(s1['F1'])} versus "
                 f"{f4(s0['F1'])}. Because F1 is the predefined primary metric, "
                 f"{chosen} class weighting was selected.\n")
        L.append(f"![Class-weight ablation]({ctx['fig_abl']})\n")


    br = ctx["bin_channel_runs"]
    if len(br) > 1:
        L.append(f"## Channel-count study ({'balanced' if ctx['best_cw'] else 'no'} class weight)\n")
        rows = []
        for c in sorted(br):
            s = br[c]["summary"]
            rows.append([c, f4(s.loc["F1", "Mean"]), f4(s.loc["Accuracy", "Mean"]),
                         f4(s.loc["AUROC", "Mean"]), f4(s.loc["F1", "Standard deviation"]),
                         f"{br[c]['seconds'] / 60:.1f}"])
        L.append(md_table(["Channels", "F1", "Accuracy", "AUROC", "F1 standard deviation", "Minutes"],
                          rows, ["r"] * 6, bold_cols=[1, 2, 3], bold_min_cols=[4]) + "\n")
        L.append(f"The {ctx['best_ch']}-channel configuration achieved the highest mean F1 "
                 f"({f4(br[ctx['best_ch']]['summary'].loc['F1', 'Mean'])}) and was retained as "
                 f"the final binary TCN.\n")
        L.append(f"![Channel study]({ctx['fig_ch']})\n")


    L.append("## Final five-class cross-validation\n")
    L.append(f"Five-class model: same architecture, {fm['cfg']['channels']} channels, "
             "cross-entropy loss (classes are balanced, so no class weight).\n")
    L.append(summary_table(fm, MULTI_METRICS, "Macro F1") + "\n")
    L.append("The fold-level Macro F1 scores were "
             + ", ".join(f4(v) for v in fm["fold_df"]["Macro F1"]) + ".\n")
    L.append(f"![Five-class fold metrics]({ctx['fig_multi_folds']})\n")

    mr = ctx["multi_channel_runs"]
    if len(mr) > 1:
        L.append("### Five-class channel-count study\n")
        rows = []
        for c in sorted(mr):
            s = mr[c]["summary"]
            rows.append([c, f4(s.loc["Accuracy", "Mean"]), f4(s.loc["Macro F1", "Mean"]),
                         f4(s.loc["Macro ROC-AUC (one-vs-rest)", "Mean"]),
                         f4(s.loc["Macro F1", "Standard deviation"])])
        L.append(md_table(["Channels", "Accuracy", "Macro F1", "Macro ROC-AUC", "Macro F1 standard deviation"],
                          rows, ["r"] * 5, bold_cols=[1, 2, 3], bold_min_cols=[4]) + "\n")


    L.append("## Five-class error analysis\n")
    L.append("Per-class scores are computed on the pooled out-of-fold predictions "
             "(each segment predicted by the model that did not train on it).\n")
    p, r, f, _ = ctx["per_class"]
    rows = [[c, CLASS_NAMES[c], f4(p[c - 1]), f4(r[c - 1]), f4(f[c - 1])] for c in range(1, 6)]
    L.append(md_table(["Original class", "Recording condition", "Precision", "Recall", "F1"],
                      rows, ["r", "l", "r", "r", "r"], bold_cols=[4]) + "\n")
    pairs = ", ".join(f"class {a} predicted as class {b} ({n} segments)" for a, b, n in ctx["top_pairs"])
    L.append(f"The most frequent errors were {pairs}. "
             f"Seizure activity (class 1) achieved F1 = {f4(f[0])}.\n")
    L.append(f"![Five-class confusion matrix]({ctx['fig_multi_cm']})\n")
    L.append(f"![Example signals]({ctx['fig_examples']})\n")


    if EVALUATE_FINAL_TEST:
        L.append("## Final test set (ensemble of the five fold models)\n")
        rows = [[m, f4(v)] for m, v in ctx["test_metrics"].items()]
        L.append(md_table(["Metric", "Value"], rows, ["l", "r"]) + "\n")
        L.append(f"![Final test curves]({ctx['fig_test']})\n")


    L.append("## Model interpretation, strengths, and limitations\n")
    L.append("The TCN stacks dilated causal 1-D convolutions with residual connections. Each "
             "block doubles the dilation, so a small number of layers covers the whole segment "
             f"(receptive field {receptive_field()} samples). The convolution filters are learned "
             "end-to-end from the raw signal together with the classifier.\n")
    L.append("Strengths: learned temporal filters, no hand-crafted time- or frequency-domain "
             "features, a compact model, and direct use of raw ordered EEG samples. "
             "Limitations: results depend on stochastic training (initialisation, batch order, "
             "early stopping), "
             "learned filters are not directly interpretable clinically, and the supplied CSV "
             "contains no reliable subject or recording identifiers, so the shared split is "
             "stratified at segment level rather than subject level. This limitation must be "
             "disclosed.\n")


    L.append("## Reproducibility notes\n")
    L.append("All values above were generated automatically by `tcn_seizure.py` from the CSV "
             "files in `results/` (`cv_<task>_<config>.csv`, one row per fold). Standardisation "
             "statistics, early-stopping splits and model fitting are computed inside each "
             "training fold only. Standard deviations are sample SDs (ddof = 1) across the five "
             "folds. " + ("The final-test partition was evaluated once after model selection."
                          if EVALUATE_FINAL_TEST else
                          "The locked final-test partition was not read during these experiments.")
             + "\n")

    path = os.path.join(OUTPUT_DIR, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


FOLD_OF_DEV = None


def save_history(res, out_dir):
    rows, runtime = [], []
    for k, h in enumerate(res["histories"]):
        for e, (tl, el) in enumerate(zip(h["train_loss"], h["es_loss"]), start=1):
            rows.append({"fold": k, "epoch": e, "train_loss": tl, "early_stop_loss": el,
                         "selected_epoch": int(e == h["best_epoch"])})
        runtime.append({"fold": k, "epochs_trained": len(h["train_loss"]),
                        "best_epoch": h["best_epoch"], "seconds": round(h.get("seconds", float("nan")), 1)})
    runtime.append({"fold": "total", "epochs_trained": sum(r["epochs_trained"] for r in runtime),
                    "best_epoch": "", "seconds": round(res["seconds"], 1)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    pd.DataFrame(runtime).to_csv(os.path.join(out_dir, "runtime.csv"), index=False)


GROUP_METRIC_NAMES = {
    "Accuracy": "accuracy", "Balanced accuracy": "balanced_accuracy", "Precision": "precision",
    "Recall": "recall", "Specificity": "specificity", "F1": "f1", "AUROC": "roc_auc",
    "Average precision": "pr_auc", "Macro precision": "macro_precision", "Macro recall": "macro_recall",
    "Macro F1": "macro_f1", "Macro ROC-AUC (one-vs-rest)": "macro_roc_auc",
}


def save_group_format(res, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    binary = res["task"] == "binary"
    prefix = "tcn_binary" if binary else "tcn_five_class"
    yt, yp = res["test_true"], res["test_prob"]

    pd.DataFrame([{GROUP_METRIC_NAMES[k]: v for k, v in metrics.items()}]).to_csv(
        os.path.join(out_dir, f"{prefix}_final_metrics.csv"), index=False)

    if binary:
        pred = pd.DataFrame({"row_index": res["test_idx"], "y_true": yt,
                             "y_pred": (yp >= THRESHOLD).astype(int), "prob_1": yp})
    else:
        pred = pd.DataFrame({"row_index": res["test_idx"], "y_true": yt + 1, "y_pred": yp.argmax(1) + 1,
                             **{f"prob_{c}": yp[:, c - 1] for c in range(1, 6)}})
    pred.to_csv(os.path.join(out_dir, f"{prefix}_final_test_predictions.csv"), index=False)

    pd.DataFrame({"fold": range(len(res["histories"])),
                  "best_epoch": [h["best_epoch"] for h in res["histories"]],
                  "train_seconds": [h.get("seconds", float("nan")) for h in res["histories"]]}
                 ).to_csv(os.path.join(out_dir, f"{prefix}_final_training_log.csv"), index=False)


def save_task_outputs(res, fold, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    binary = res["task"] == "binary"
    dev_idx, test_idx = res["dev_idx"], res["test_idx"]

    def to_label(y):
        return y if binary else y + 1

    def pred_of(prob):
        return (prob >= THRESHOLD).astype(int) if binary else prob.argmax(1)

    def prob_cols(prob):
        if binary:
            return {"y_prob": prob}
        return {f"prob_class_{c}": prob[:, c - 1] for c in range(1, 6)}

    pd.DataFrame({"row_index": dev_idx, "cv_fold": fold[dev_idx],
                  "y_true": to_label(res["oof_true"]),
                  "y_pred": to_label(pred_of(res["oof_prob"])),
                  **prob_cols(res["oof_prob"])}
                 ).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)

    save_history(res, out_dir)

    metrics = None
    if res["test_prob"] is not None:
        yt, yp = res["test_true"], res["test_prob"]
        metrics = binary_metrics(yt, yp) if binary else multi_metrics(yt, yp)
        pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "final_metrics.csv"), index=False)
        pd.DataFrame({"row_index": test_idx, "y_true": to_label(yt), "y_pred": to_label(pred_of(yp))}
                     ).to_csv(os.path.join(out_dir, "final_predictions.csv"), index=False)
        pd.DataFrame({"row_index": test_idx, **prob_cols(yp)}
                     ).to_csv(os.path.join(out_dir, "final_probabilities.csv"), index=False)

        labels = [0, 1] if binary else list(range(5))
        names = ["Non-seizure", "Seizure"] if binary else [str(c) for c in range(1, 6)]
        cm = confusion_matrix(yt, pred_of(yp), labels=labels)
        pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]
                     ).to_csv(os.path.join(out_dir, "confusion_matrix.csv"))
        title = "Binary" if binary else "Five-class"
        plot_confusion(cm, names, f"{title} confusion matrix (final test)", "confusion_matrix.png",
                       figsize=(4.6, 4) if binary else (6, 5.2), folder=out_dir)

        save_group_format(res, metrics, os.path.join(OUTPUT_DIR, "update"))

    print(f"Saved outputs in: {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        print(f"    {f}")
    return metrics


def write_readme(results, path):
    b, m = results["binary"], results["multiclass"]
    L = ["# TCN final outputs", ""]
    L += ["## Configuration", "",
          f"- Binary task: `y == 1` (seizure) versus `y in {{2, 3, 4, 5}}`; configuration `{b['cfg']['name']}` "
          f"({b['cfg']['channels']} channels, class weight: {'balanced' if b['cfg']['class_weight'] else 'none'})",
          f"- Five-class task: original labels 1-5; configuration `{m['cfg']['name']}` "
          f"({m['cfg']['channels']} channels, cross-entropy loss, no class weight)",
          f"- Architecture: TCN, {NUM_LEVELS} residual blocks, dilations 1-{2 ** (NUM_LEVELS - 1)}, "
          f"kernel size {KERNEL_SIZE}, dropout {DROPOUT}, receptive field {receptive_field()} samples",
          f"- Training: Adam (lr {LEARNING_RATE}), batch size {BATCH_SIZE}, up to {MAX_EPOCHS} epochs, "
          f"early stopping patience {PATIENCE} on a stratified {int(EARLY_STOP_FRACTION * 100)}% split of the training folds",
          f"- Input standardised with the mean and SD of each training fold",
          f"- Random seed: {SEED}; device: `{DEVICE}`",
          "- Split: `shared_split_with_cv.csv` (development folds 0-4 for cross-validation, `final_test` for evaluation)",
          "- Final-test predictions are the average probability of the five fold models",
          f"- Binary decision threshold: {THRESHOLD}; five-class prediction: class with the highest probability",
          ""]
    L += ["## Files", "",
          "Each of `binary_final_test/` and `five_class_final_test/` contains:", "",
          "| File | Contents |", "|---|---|",
          "| `oof_predictions.csv` | Out-of-fold predictions for the 9,200 development segments (`row_index`, `cv_fold`, `y_true`, `y_pred`, probabilities) |",
          "| `final_metrics.csv` | Metrics on the 2,300 final-test segments |",
          "| `final_predictions.csv` | Predicted label per final-test segment (`row_index`, `y_true`, `y_pred`) |",
          "| `final_probabilities.csv` | Predicted probability per final-test segment (binary: `y_prob` = P(seizure); five-class: `prob_class_1` ... `prob_class_5`) |",
          "| `confusion_matrix.csv` / `.png` | Final-test confusion matrix (rows = true, columns = predicted) |",
          "| `training_history.csv` | Train and early-stopping loss per fold and epoch; `selected_epoch = 1` marks the restored weights |",
          "| `runtime.csv` | Epochs trained, selected epoch and wall-clock seconds per fold, plus total |",
          "",
          "Binary labels: 1 = seizure, 0 = non-seizure. Five-class labels use the original dataset coding "
          "(1 seizure, 2 tumour region, 3 healthy region of a tumour patient, 4 eyes closed, 5 eyes open).",
          ""]
    L += ["## Results", ""]
    for key, res, title in [("binary", b, "Binary"), ("multiclass", m, "Five-class")]:
        L += [f"### {title}", "", "| Metric | CV mean | CV SD | Final test |", "|---|---:|---:|---:|"]
        for metric in res["summary"].index:
            L.append(f"| {metric} | {res['summary'].loc[metric, 'Mean']:.4f} | "
                     f"{res['summary'].loc[metric, 'Standard deviation']:.4f} | {results[key + '_test'][metric]:.4f} |")
        L += ["", f"Runtime: {res['seconds'] / 60:.1f} minutes for five folds.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"Saved: {path}")


def run_final_only(X, y_raw, split, fold):
    global EVALUATE_FINAL_TEST
    EVALUATE_FINAL_TEST = True
    results = {}
    for task, cfg, folder in [
        ("binary", make_cfg(FINAL_CHANNELS, FINAL_CLASS_WEIGHT), "binary_final_test"),
        ("multiclass", make_cfg(FINAL_MULTI_CHANNELS, False), "five_class_final_test"),
    ]:
        res = run_cv(task, cfg, X, y_raw, split, fold)
        results[task] = res
        results[task + "_test"] = save_task_outputs(res, fold, os.path.join(OUTPUT_DIR, folder))
        print(f"\n{task} final test:")
        for k, v in results[task + "_test"].items():
            print(f"  {k:<28s}: {v:.4f}")
    write_readme(results, os.path.join(OUTPUT_DIR, "README_final_outputs.md"))


def main():
    global FOLD_OF_DEV
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("Note: no GPU detected, the full experiment will be slow. Set QUICK_MODE = True for a quick trial run.")

    X, y_raw, split, fold = load_data()
    FOLD_OF_DEV = fold[split == "development"]
    ctx = {}

    if FINAL_ONLY:
        run_final_only(X, y_raw, split, fold)
        return


    weights = [True] if QUICK_MODE else [False, True]
    ablation = {cw: run_cv("binary", make_cfg(DEFAULT_CHANNELS, cw), X, y_raw, split, fold)
                for cw in weights}
    best_cw = max(ablation, key=lambda cw: ablation[cw]["summary"].loc["F1", "Mean"])


    grid = [DEFAULT_CHANNELS] if QUICK_MODE else CHANNEL_GRID
    bin_runs = {DEFAULT_CHANNELS: ablation[best_cw]}
    for ch in grid:
        if ch not in bin_runs:
            bin_runs[ch] = run_cv("binary", make_cfg(ch, best_cw), X, y_raw, split, fold)
    best_ch = max(bin_runs, key=lambda c: bin_runs[c]["summary"].loc["F1", "Mean"])
    final_bin = bin_runs[best_ch]


    multi_runs = {ch: run_cv("multiclass", make_cfg(ch, False), X, y_raw, split, fold)
                  for ch in grid}
    best_mch = max(multi_runs, key=lambda c: multi_runs[c]["summary"].loc["Macro F1", "Mean"])
    final_multi = multi_runs[best_mch]


    all_rows = []
    for r in list(ablation.values()) + [v for k, v in bin_runs.items() if k != DEFAULT_CHANNELS] \
            + list(multi_runs.values()):
        row = {"task": r["task"], "config": r["cfg"]["name"], "minutes": r["seconds"] / 60}
        for m in r["summary"].index:
            row[f"{m} mean"] = r["summary"].loc[m, "Mean"]
            row[f"{m} sd"] = r["summary"].loc[m, "Standard deviation"]
        all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(os.path.join(OUTPUT_DIR, "all_experiments_summary.csv"), index=False)


    print("\nGenerating figures and report...")
    bin_pred = (final_bin["oof_prob"] >= THRESHOLD).astype(int)
    ctx["bin_cm"] = confusion_matrix(final_bin["oof_true"], bin_pred, labels=[0, 1])
    m_pred = final_multi["oof_prob"].argmax(1)
    m_true = final_multi["oof_true"]
    mcm = confusion_matrix(m_true, m_pred, labels=range(5))
    ctx["per_class"] = precision_recall_fscore_support(m_true, m_pred, labels=range(5), zero_division=0)
    off = [(i + 1, j + 1, int(mcm[i, j])) for i in range(5) for j in range(5) if i != j]
    ctx["top_pairs"] = sorted(off, key=lambda t: -t[2])[:4]

    ctx.update({
        "final_bin": final_bin, "final_multi": final_multi, "ablation": ablation,
        "best_cw": best_cw, "bin_channel_runs": bin_runs, "best_ch": best_ch,
        "multi_channel_runs": multi_runs,
        "fig_examples": plot_example_signals(X, y_raw, split),
        "fig_roc": plot_binary_curves(final_bin),
        "fig_bin_folds": plot_fold_metrics(final_bin, BINARY_METRICS, "binary_fold_metrics.png",
                                           "Binary TCN: metrics per validation fold"),
        "fig_bin_cm": plot_confusion(ctx["bin_cm"], ["Non-seizure", "Seizure"],
                                     "Binary confusion matrix (pooled CV)", "binary_confusion_matrix.png"),
        "fig_train": plot_training_curves(final_bin, "binary_training_curves.png"),
        "fig_multi_folds": plot_fold_metrics(final_multi, MULTI_METRICS, "multiclass_fold_metrics.png",
                                             "Five-class TCN: metrics per validation fold"),
        "fig_multi_cm": plot_confusion(mcm, [f"{c}" for c in range(1, 6)],
                                       "Five-class confusion matrix (pooled CV)",
                                       "multiclass_confusion_matrix.png", figsize=(6, 5.2)),
    })
    if len(ablation) > 1:
        ctx["fig_abl"] = plot_config_bars(
            [ablation[False], ablation[True]], ["No class weight", "Balanced class weight"],
            ["F1", "Precision", "Recall", "Specificity"], "ablation_class_weight.png",
            "Class-weight ablation")
    if len(bin_runs) > 1:
        ctx["fig_ch"] = plot_channel_study(bin_runs, multi_runs, "channel_study.png")


    if EVALUATE_FINAL_TEST:
        yt, yp = final_bin["test_true"], final_bin["test_prob"]
        ctx["test_metrics"] = binary_metrics(yt, yp)
        pd.DataFrame([ctx["test_metrics"]]).to_csv(os.path.join(OUTPUT_DIR, "final_test_results.csv"), index=False)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        fpr, tpr, _ = roc_curve(yt, yp)
        pr, rc, _ = precision_recall_curve(yt, yp)
        ax[0].plot(fpr, tpr, color=SERIES[0], label=f"AUROC {ctx['test_metrics']['AUROC']:.4f}")
        ax[0].plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
        ax[1].plot(rc, pr, color=SERIES[0], label=f"AP {ctx['test_metrics']['Average precision']:.4f}")
        ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (final test)")
        ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall (final test)")
        ax[0].legend(); ax[1].legend()
        ctx["fig_test"] = savefig(fig, "final_test_curves.png")

    save_task_outputs(final_bin, fold, os.path.join(OUTPUT_DIR, "binary_final_test"))
    save_task_outputs(final_multi, fold, os.path.join(OUTPUT_DIR, "five_class_final_test"))
    path = write_report(ctx)


    print("\n" + "=" * 60)
    print(f"Final binary model: {final_bin['cfg']['name']}")
    print(final_bin["summary"].round(4).to_string())
    print(f"\nFinal five-class model: {final_multi['cfg']['name']}")
    print(final_multi["summary"].round(4).to_string())
    print("=" * 60)
    print(f"Report saved: {path}")
    print(f"Figures saved in: {FIG_DIR}/")


    import shutil
    zip_path = shutil.make_archive("tcn_results", "zip", OUTPUT_DIR)
    print(f"Archive saved: {zip_path}")


if __name__ == "__main__":
    main()
