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


def _complete_seed_dirs(exp_dir):
    out = {}
    for d in sorted(glob.glob(os.path.join(exp_dir, "seed*"))):
        if len(glob.glob(os.path.join(d, "preds_fold*.csv"))) == config.N_FOLDS:
            out[int(os.path.basename(d)[4:])] = d
    return out


def all_fold_metrics(runs_dir):
    """One row per (experiment, seed, fold) for every complete run under runs_dir/<experiment>/seed<seed>/."""
    rows = []
    for exp_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if not os.path.isdir(exp_dir):
            continue
        for seed, d in _complete_seed_dirs(exp_dir).items():
            for k, p in load_run(d).items():
                m = fold_metrics(p)
                rows.append({"experiment": os.path.basename(exp_dir), "seed": seed, "fold": k,
                             **{x: m[x] for x in METRICS}})
    return pd.DataFrame(rows)


def summarize_runs(runs_dir):
    """Mean ± std across seeds (each seed = mean over its folds), for every experiment."""
    R = all_fold_metrics(runs_dir)
    if R.empty:
        print(f"No complete runs found in {runs_dir}")
        return R
    seed_means = R.groupby(["experiment", "seed"])[METRICS].mean()
    summary = seed_means.groupby("experiment").agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "n_seeds", seed_means.groupby("experiment").size())
    summary.to_csv(os.path.join(runs_dir, "summary_across_seeds.csv"))
    R.to_csv(os.path.join(runs_dir, "all_fold_metrics.csv"), index=False)
    print("\nMean ± std across seeds:")
    for exp, r in summary.iterrows():
        print(f"  {exp:<18} ({int(r.n_seeds)} seeds) " +
              " | ".join(f"{m} {100 * r[m + '_mean']:.2f} ± {100 * r[m + '_std']:.2f}" for m in METRICS))
    return summary


def ablation_report(runs_dir, out_md=None):
    """Ablation tables (mean ± std across seeds) with Wilcoxon signed-rank tests against the reference.

    The test pairs the per-(seed, fold) results of two experiments; this is valid because every
    experiment is evaluated on exactly the same outer test folds.
    """
    from scipy.stats import wilcoxon

    from .experiments import ABLATION_REFERENCE, ABLATION_TABLES, DISPLAY_NAMES

    R = all_fold_metrics(runs_dir)
    have = set(R["experiment"]) if not R.empty else set()
    if ABLATION_REFERENCE not in have:
        raise RuntimeError(f"Reference experiment '{ABLATION_REFERENCE}' has no complete runs in {runs_dir}")
    S = R.groupby(["experiment", "seed"])[METRICS].mean().groupby("experiment").agg(["mean", "std"]) * 100
    ref = R[R.experiment == ABLATION_REFERENCE].set_index(["seed", "fold"])

    def p_value(name, metric):
        if name == ABLATION_REFERENCE:
            return "-"
        a = R[R.experiment == name].set_index(["seed", "fold"])[metric]
        b = ref[metric].reindex(a.index)
        if np.allclose(a.values, b.values):
            return "1.000"
        return f"{wilcoxon(a.values, b.values, zero_method='zsplit').pvalue:.3f}"

    lines = []
    for title, names in ABLATION_TABLES.items():
        names = [n for n in names if n in have]
        lines += [f"**{title}**", "",
                  "| Model | " + " | ".join(METRICS) + " | p (Acc) | p (AUC) |",
                  "|---" * (len(METRICS) + 3) + "|"]
        for n in names:
            label = DISPLAY_NAMES.get(n, n)
            label = f"**{label}**" if n == ABLATION_REFERENCE else label
            cells = " | ".join(f"{S.loc[n, (m, 'mean')]:.2f} ± {S.loc[n, (m, 'std')]:.2f}" for m in METRICS)
            lines.append(f"| {label} | {cells} | {p_value(n, 'Accuracy')} | {p_value(n, 'AUC')} |")
        lines.append("")
    md = "\n".join(lines)
    print(md)
    out_md = out_md or os.path.join(runs_dir, "ablation_tables.md")
    with open(out_md, "w") as f:
        f.write(md)
    print(f"Saved {out_md}")
    return md


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
