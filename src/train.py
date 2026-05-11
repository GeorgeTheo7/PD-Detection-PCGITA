# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import os
import random
from . import config
from .dataset import PCGitaDataset
from .models import ParkinsonCoAttentionModel

#Nested Cross-Validation (Outer 5-Fold K-Fold ανά ασθενή, Inner 15% Split για Early Stopping)

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Train / Validate helpers
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader: #moves data to GPU
        s = batch["spectrogram"].to(device)
        m = batch["mfcc"].to(device)
        w = batch["wav2vec"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        #input goes through all three branches --> co-attention --> classifier
        optimizer.zero_grad(set_to_none=True)
        outputs = model(s, m, w, mask) 
        loss = criterion(outputs, labels) #computes the difference between predicted logits and true labels

        loss.backward() #computes gradients of loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #prevents exploding gradients
        optimizer.step() #Adam optimizer updates model weights

        running_loss += loss.item() * s.size(0)

    return running_loss / len(loader.dataset) #return average loss over epoch

def validate(model, loader, device):
    """
    Returns per-chunk predictions with subject_id preserved.
    """
    model.eval()
    val_results = {"subject_id": [], "prob_pd": [], "label": []}

    with torch.no_grad():
        for batch in loader:
            s = batch["spectrogram"].to(device)
            m = batch["mfcc"].to(device)
            w = batch["wav2vec"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(s, m, w, mask)
            probs = torch.softmax(outputs, dim=1)[:, 1]  #convert logits to PD probabilities

            subject_ids = batch["subject_id"]  #list[str]

            val_results["prob_pd"].extend(probs.cpu().numpy().tolist())
            val_results["label"].extend(labels.cpu().numpy().tolist())
            val_results["subject_id"].extend(subject_ids)

    return pd.DataFrame(val_results) #DataFrame with per-chunk predictions

# Main training loop (Outer CV + Inner Early-Stop split)
def run_training():
    set_seed(config.SEED) #for reproducibility
    print(f"Loading data from {config.CSV_PATH}...") 
    dataset = PCGitaDataset(config.CSV_PATH, config.CHUNKS_DIR) #load dataset

    # Outer split uses per-row labels + per-row subject groups
    row_subjects = dataset.data["subject_id"].astype(str).tolist() #list of subject IDs for each chunk
    row_labels = dataset.data["label"].tolist() #list of labels for each chunk

    outer_gkf = StratifiedGroupKFold(n_splits=config.N_FOLDS) #stratified group k-fold split
    #maintains class balance in each fold and ensures no subject leakage across folds

    # Inner split configuration (ONLY for early stopping / best epoch)
    INNER_ES_FRAC = 0.15     #15% of training subjects for early stopping
    PATIENCE = 3             #stop if F1 doesn't improve for 3 epochs

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        outer_gkf.split(np.zeros(len(dataset)), row_labels, groups=row_subjects)
    ): #for each outer fold, trainval_idx --> indices for training+validation, test_idx --> indices for testing (20% of data)
        print(f"\n=== OUTER FOLD {fold_idx+1}/{config.N_FOLDS} ===")

        # Build subject-level table from OUTER trainval split
        trainval_df = dataset.data.iloc[trainval_idx].copy() #only rows in trainval split
        trainval_df["subject_id"] = trainval_df["subject_id"].astype(str)

        subj_df = trainval_df.groupby("subject_id", as_index=False)["label"].first() #group chunks by subject_id, get first label (all chunks from same subject have same label)
        subj_ids = subj_df["subject_id"].values
        subj_y = subj_df["label"].values

        #Inner stratified subject split: train_inner vs early_stop_inner
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=INNER_ES_FRAC,
            random_state=config.SEED + fold_idx
        )
        train_subj_idx, es_subj_idx = next(sss.split(subj_ids, subj_y))
        train_subj_set = set(subj_ids[train_subj_idx])
        es_subj_set = set(subj_ids[es_subj_idx])

        # Map subject sets back to ROW indices (indices must refer to dataset rows)
        train_inner_idx = trainval_df.index[trainval_df["subject_id"].isin(train_subj_set)].tolist()
        es_inner_idx = trainval_df.index[trainval_df["subject_id"].isin(es_subj_set)].tolist()
        test_idx = list(test_idx)

        # Sanity checks: no subject leakage across splits
        train_subjects = set(dataset.data.iloc[train_inner_idx]["subject_id"].astype(str))
        es_subjects = set(dataset.data.iloc[es_inner_idx]["subject_id"].astype(str))
        test_subjects = set(dataset.data.iloc[test_idx]["subject_id"].astype(str))

        assert train_subjects.isdisjoint(es_subjects), "Leakage: train_inner vs early_stop_inner share subjects!"
        assert train_subjects.isdisjoint(test_subjects), "Leakage: train_inner vs outer_test share subjects!"
        assert es_subjects.isdisjoint(test_subjects), "Leakage: early_stop_inner vs outer_test share subjects!"

        print(f"Subjects | train_inner: {len(train_subjects)} | inner_ES: {len(es_subjects)} | outer_test: {len(test_subjects)}")

        # DataLoaders
        train_loader = DataLoader(
            Subset(dataset, train_inner_idx),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

        es_loader = DataLoader(
            Subset(dataset, es_inner_idx),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

        test_loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

        #Model / Optim
        model = ParkinsonCoAttentionModel().to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        #Track best epoch according to INNER early-stop F1
        best_f1 = -1.0
        bad_epochs = 0

        fold_ckpt_path = os.path.join(config.SAVE_MODEL_DIR, f"model_fold{fold_idx}_best.pth")
        fold_pred_path = os.path.join(config.SAVE_MODEL_DIR, f"preds_fold{fold_idx}.csv")

        for epoch in range(config.N_EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)

            #Evaluate on INNER early-stop set (subject-level aggregation)
            df_es = validate(model, es_loader, config.DEVICE)
            subj_scores = df_es.groupby("subject_id")["prob_pd"].mean()
            subj_labels = df_es.groupby("subject_id")["label"].first()
            subj_preds = (subj_scores > 0.5).astype(int)

            f1 = f1_score(subj_labels, subj_preds, zero_division=0)
            print(f"Ep {epoch+1:02d} | Loss: {loss:.4f} | INNER-ES F1: {f1:.4f}")

            #Save best checkpoint based ONLY on inner ES (not outer test)
            if f1 > best_f1:
                best_f1 = f1
                bad_epochs = 0
                torch.save(model.state_dict(), fold_ckpt_path)
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}. Best INNER-ES F1 = {best_f1:.4f}")
                    break

        #Load best epoch and evaluate ONCE on OUTER TEST fold
        if os.path.exists(fold_ckpt_path):
            model.load_state_dict(torch.load(fold_ckpt_path, map_location=config.DEVICE)) #loads the weights from the epoch with best INNER-ES F1
        else:
            print("WARNING: No checkpoint found; using last epoch weights.")

        df_test = validate(model, test_loader, config.DEVICE)
        df_test.to_csv(fold_pred_path, index=False)
        print(f"✅ Saved OUTER test predictions -> {fold_pred_path}")

if __name__ == "__main__":
    run_training()
