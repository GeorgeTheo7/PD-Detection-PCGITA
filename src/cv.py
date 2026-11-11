"""Speaker-independent nested cross-validation.

Outer loop : 5 folds, every speaker's chunks (both tasks) in exactly one test fold.
             The fold assignment is fixed in splits/pcgita_outer_folds.csv for reproducibility.
Inner split: 15% of the training speakers (stratified) are held out for early stopping only.
Prediction : chunk probabilities are averaged per speaker (subject-level diagnosis).
"""
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from . import config
from .evaluate import subject_level
from .models import FusionModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def outer_folds(df, folds_csv=None):
    """Yield (fold_number, trainval_row_idx, test_row_idx)."""
    subj = df["subject_id"].astype(str).values
    if folds_csv and os.path.exists(folds_csv):
        fmap = pd.read_csv(folds_csv, dtype={"subject_id": str}).set_index("subject_id")["fold"]
        missing = set(subj) - set(fmap.index)
        if missing:
            raise ValueError(f"{len(missing)} speakers are not in {folds_csv}, e.g. {sorted(missing)[:3]}")
        fold_of_row = fmap.loc[subj].values
        for k in sorted(set(fold_of_row)):
            yield k, np.where(fold_of_row != k)[0], np.where(fold_of_row == k)[0]
    else:
        gkf = StratifiedGroupKFold(n_splits=config.N_FOLDS)
        for f, (tr, te) in enumerate(gkf.split(np.zeros(len(df)), df["label"], groups=subj)):
            yield f + 1, tr, te


class FeatureStore:
    """All features in memory (CPU); batches are moved to the GPU on demand.

    ssl: dict {ssl_name: tensor [N, T, D]} with the cached frozen SSL features that are needed.
    """

    def __init__(self, df, spec, mfcc, ssl, device):
        self.df, self.spec, self.mfcc, self.ssl, self.device = df, spec, mfcc, ssl, device
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

    def batch(self, idx, cfg):
        t = torch.from_numpy(np.asarray(idx))
        s = self.spec[t].float().to(self.device) if cfg["use_spec"] else None
        m = self.mfcc[t].to(self.device) if cfg["use_mfcc"] else None
        w = self.ssl[cfg["ssl"]][t].float().to(self.device) if cfg["ssl"] else None
        return s, m, w, self.labels[t].to(self.device)


def build_model(cfg, device):
    ssl_dim = config.SSL_MODELS[cfg["ssl"]]["dim"] if cfg["ssl"] else 768
    return FusionModel(cfg["fusion"], cfg["use_spec"], cfg["use_mfcc"], ssl_dim).to(device)


@torch.no_grad()
def predict(model, store, idx, cfg):
    model.eval()
    probs = []
    for i in range(0, len(idx), config.BATCH_SIZE):
        s, m, w, _ = store.batch(idx[i:i + config.BATCH_SIZE], cfg)
        probs.append(torch.softmax(model(s, m, w), dim=1)[:, 1].cpu())
    out = store.df.iloc[idx][["filename", "subject_id", "task", "label"]].copy()
    out["prob_pd"] = torch.cat(probs).numpy()
    return out


def early_stopping_score(preds, mode):
    """Higher is better. 'f1' = subject-level F1 (thesis protocol); 'loss' = -chunk-level cross-entropy."""
    if mode == "f1":
        y, _, pred = subject_level(preds)
        return f1_score(y, pred, zero_division=0)
    if mode == "loss":
        p = preds["prob_pd"].clip(1e-7, 1 - 1e-7).values
        y = preds["label"].values
        return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    raise ValueError(f"Unknown early-stopping mode: {mode}")


def run_cv(store, out_dir, cfg, seed=42, es_mode="f1", folds_csv=None, verbose=True, tag=""):
    """Train/evaluate all outer folds of one experiment and seed.

    cfg: experiment config from src/experiments.py. Skips folds whose predictions already exist (resumable).
    """
    os.makedirs(out_dir, exist_ok=True)
    df = store.df
    subj = df["subject_id"].astype(str).values

    for k, trval, test in outer_folds(df, folds_csv):
        pred_path = os.path.join(out_dir, f"preds_fold{k}.csv")
        if os.path.exists(pred_path):
            print(f"[{tag} seed {seed} | {es_mode}] fold {k}: already done, skipping")
            continue

        fold_seed = seed + k - 1
        set_seed(fold_seed)

        # Inner, speaker-level split of the training speakers for early stopping
        spk = df.iloc[trval].groupby("subject_id")["label"].first()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=config.ES_FRAC, random_state=fold_seed)
        tr_s, es_s = next(sss.split(spk.index.values, spk.values))
        tr_set, es_set = set(spk.index.values[tr_s]), set(spk.index.values[es_s])
        tr_idx = trval[np.isin(subj[trval], list(tr_set))]
        es_idx = trval[np.isin(subj[trval], list(es_set))]
        te_set = set(subj[test])
        assert tr_set.isdisjoint(es_set) and tr_set.isdisjoint(te_set) and es_set.isdisjoint(te_set), \
            "Speaker leakage between splits!"
        if verbose:
            print(f"\n[{tag} seed {seed} | {es_mode}] FOLD {k} | speakers train={len(tr_set)} "
                  f"early-stop={len(es_set)} test={len(te_set)}")

        model = build_model(cfg, store.device)
        opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        crit = nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(fold_seed)

        best, best_ep, bad, best_state = -np.inf, 0, 0, None
        for ep in range(1, config.MAX_EPOCHS + 1):
            t0 = time.time()
            model.train()
            total = 0.0
            order = tr_idx[torch.randperm(len(tr_idx), generator=gen).numpy()]
            for i in range(0, len(order), config.BATCH_SIZE):
                s, m, w, y = store.batch(order[i:i + config.BATCH_SIZE], cfg)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(s, m, w), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
                opt.step()
                total += loss.item() * len(y)
            score = early_stopping_score(predict(model, store, es_idx, cfg), es_mode)
            if verbose:
                print(f"  ep {ep:02d} | train loss {total / len(tr_idx):.4f} | "
                      f"early-stop {es_mode} {score:.4f} | {time.time() - t0:.0f}s")
            if score > best:
                best, best_ep, bad = score, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= config.PATIENCE:
                    break

        # Evaluate the best epoch ONCE on the untouched outer test fold
        model.load_state_dict(best_state)
        preds = predict(model, store, test, cfg)
        preds.to_csv(pred_path, index=False)

        y, score_s, pred_s = subject_level(preds)
        log = {"fold": k, "seed": seed, "es_mode": es_mode, "best_epoch": best_ep, "best_es_score": best,
               "test_accuracy": accuracy_score(y, pred_s), "test_recall": recall_score(y, pred_s),
               "test_auc": roc_auc_score(y, score_s)}
        with open(os.path.join(out_dir, f"log_fold{k}.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"[{tag} seed {seed} | {es_mode}] fold {k}: best epoch {best_ep} | test acc {log['test_accuracy']:.2f} "
              f"| recall {log['test_recall']:.2f} | AUC {log['test_auc']:.3f}")
