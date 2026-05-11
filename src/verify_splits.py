import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import src.config as config  # Ensure this points to your config file


def verify_splits():
    print(f"Loading data from: {config.CSV_PATH}")
    df = pd.read_csv(config.CSV_PATH)

    # 1. Setup Data
    subject_ids = df['subject_id'].values
    labels = df['label'].values

    # 2. Setup Splitter (Exact same logic as train.py)
    gkf = StratifiedGroupKFold(n_splits=config.N_FOLDS)

    print("\n" + "=" * 50)
    print("🚑 DATA LEAKAGE CHECK (Subject Isolation)")
    print("=" * 50)

    leakage_found = False

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, labels, groups=subject_ids)):
        # Get the actual subject IDs in Train and Val
        train_subjects = set(subject_ids[train_idx])
        val_subjects = set(subject_ids[val_idx])

        # Check for intersection
        overlap = train_subjects.intersection(val_subjects)

        # Stats
        n_train = len(train_subjects)
        n_val = len(val_subjects)
        n_overlap = len(overlap)

        print(f"Fold {fold + 1}:")
        print(f"   Train Patients: {n_train}")
        print(f"   Val Patients:   {n_val}")

        if n_overlap > 0:
            print(f"   ⚠️ LEAKAGE DETECTED! {n_overlap} subjects are in both sets.")
            print(f"   Leaking IDs: {overlap}")
            leakage_found = True
        else:
            print(f"   ✅ CLEAN. (0 overlapping subjects)")

        # Optional: Check Class Balance (Stratification)
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_pd_perc = np.mean(train_labels) * 100
        val_pd_perc = np.mean(val_labels) * 100
        print(f"   Dist: Train PD={train_pd_perc:.1f}% | Val PD={val_pd_perc:.1f}%")
        print("-" * 30)

    print("=" * 50)
    if leakage_found:
        print("❌ CRITICAL FAILURE: Your splits are leaking subjects.")
    else:
        print("✅ SUCCESS: Your validation is rigorous. No subject leakage.")
    print("=" * 50)


if __name__ == "__main__":
    verify_splits()