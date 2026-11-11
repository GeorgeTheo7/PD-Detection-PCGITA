"""Subject-level evaluation: per-fold metrics, multi-seed summary, confusion matrix and ROC plots."""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from . import config

METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC"]


def subject_level(preds, threshold=config.DECISION_THRESHOLD):
    """Average chunk probabilities per speaker -> (true label, P(PD), predicted label), indexed by speaker."""
    p = preds.copy()
    p["subject_id"] = p["subject_id"].astype(str)
    score = p.groupby("subject_id")["prob_pd"].mean()
    label = p.groupby("subject_id")["label"].first()
    return label, score, (score > threshold).astype(int)


def fold_metrics(preds):
    y, score, pred = subject_level(preds)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "Specificity": tn / (tn + fp),
            "F1": f1_score(y, pred, zero_division=0), "AUC": roc_auc_score(y, score),
            "TP": tp, "FN": fn, "TN": tn, "FP": fp}


def load_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "preds_fold*.csv")))
    if not files:
        raise FileNotFoundError(f"No preds_fold*.csv in {run_dir}")
    return {int(os.path.basename(f)[len("preds_fold"):-4]): pd.read_csv(f) for f in files}


def evaluate_run(run_dir, make_plots=True):
    """Per-fold table + mean ± std for one run (one seed)."""
    preds = load_run(run_dir)
    res = pd.DataFrame([{"Fold": k, **fold_metrics(p)} for k, p in sorted(preds.items())])
    res.to_csv(os.path.join(run_dir, "per_fold_metrics.csv"), index=False)

    print(f"\n=== {run_dir} ===")
    print(res.round(4).to_string(index=False))
    print("\nMean ± std over folds:")
    for m in METRICS:
        print(f"  {m:<12} {res[m].mean():.4f} ± {res[m].std(ddof=0):.4f}")

    print("\nPer task (same models, one task's chunks per speaker):")
    for task in sorted(pd.concat(preds.values())["task"].unique()):
        acc = [fold_metrics(p[p["task"] == task])["Accuracy"] for p in preds.values()]
        auc = [fold_metrics(p[p["task"] == task])["AUC"] for p in preds.values()]
        print(f"  {task:<10} Acc {np.mean(acc):.4f} ± {np.std(acc):.4f} | AUC {np.mean(auc):.4f} ± {np.std(auc):.4f}")

    if make_plots:
        plot_confusion_matrix(preds, os.path.join(run_dir, "confusion_matrix.png"))
        plot_roc(preds, os.path.join(run_dir, "roc_curve.png"))
    return res


def summarize_runs(runs_dir):
    """Multi-seed summary: one row per run (mean over folds), then mean ± std across seeds."""
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "es_*_seed*"))):
        name = os.path.basename(run_dir)                     # es_<mode>_seed<seed>
        mode, seed = name[3:].rsplit("_seed", 1)
        preds = load_run(run_dir)
        if len(preds) < config.N_FOLDS:
            print(f"Skipping incomplete run {name} ({len(preds)}/{config.N_FOLDS} folds)")
            continue
        fm = pd.DataFrame([fold_metrics(p) for p in preds.values()])
        rows.append({"ES": mode, "Seed": int(seed), **fm[METRICS].mean().to_dict()})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(runs_dir, "multiseed_summary.csv"), index=False)
    print("\nPer seed (mean over 5 folds):")
    print(summary.round(4).to_string(index=False))
    print("\nMean ± std across seeds:")
    print(summary.groupby("ES")[METRICS].agg(["mean", "std"]).round(4).to_string())
    return summary


def plot_confusion_matrix(preds, path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    ys, ps = [], []
    for p in preds.values():
        y, _, pred = subject_level(p)
        ys += list(y)
        ps += list(pred)
    cm = confusion_matrix(ys, ps, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Parkinson's"]).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"Subject-level confusion matrix\n({len(ys)} speakers, pooled over folds)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(preds, path):
    import matplotlib.pyplot as plt

    grid = np.linspace(0, 1, 101)
    tprs, aucs = [], []
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for k, p in sorted(preds.items()):
        y, score, _ = subject_level(p)
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        ax.plot(fpr, tpr, lw=1, alpha=0.35, label=f"Fold {k} (AUC {auc:.3f})")
        t = np.interp(grid, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
        aucs.append(auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    ax.plot(grid, mean_tpr, color="green", lw=2.5, label=f"Mean (AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f})")
    ax.fill_between(grid, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1),
                    color="grey", alpha=0.2)
    ax.plot([0, 1], [0, 1], "--", color="red", alpha=0.6, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Subject-level ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
