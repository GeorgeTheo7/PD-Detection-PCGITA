import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
)
import os
from . import config

#Υπολογίζουμε τις τελικές μετρικές για όλα τα folds


def run_evaluation():

    SAVE_DIR = config.SAVE_MODEL_DIR
    N_FOLDS = config.N_FOLDS

    #Storage for metrics across folds
    metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'Specificity': [],
        'AUC': []
    }

    #Storage for ROC plotting
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    print("\n" + "=" * 40)
    print("=== STARTING EVALUATION ===")
    print("=" * 40)
    print(f"Loading predictions from: {SAVE_DIR}")

    fold_metrics_list = []

    for fold_idx in range(N_FOLDS):
        file_path = os.path.join(SAVE_DIR, f"preds_fold{fold_idx}.csv")
        if not os.path.exists(file_path):
            print(f"⚠️ Missing file: {file_path}")
            continue

        df = pd.read_csv(file_path)

        #Aggregate by Subject (Mean Probability)
        df['subject_id'] = df['subject_id'].astype(str)

        subject_scores = df.groupby('subject_id')['prob_pd'].mean()
        subject_labels = df.groupby('subject_id')['label'].first()

        #Diagnosis
        subject_preds = (subject_scores > 0.5).astype(int)

        #Calculate Metrics for this Fold
        acc = accuracy_score(subject_labels, subject_preds)
        prec = precision_score(subject_labels, subject_preds, zero_division=0)
        rec = recall_score(subject_labels, subject_preds, zero_division=0)  # Sensitivity
        f1 = f1_score(subject_labels, subject_preds, zero_division=0)

        # Calculate AUC
        try:
            fold_auc = roc_auc_score(subject_labels, subject_scores)
        except:
            fold_auc = 0.5

        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(subject_labels, subject_preds).ravel()
        spec = tn / (tn + fp + 1e-9)

        # Store
        metrics['Accuracy'].append(acc)
        metrics['Precision'].append(prec)
        metrics['Recall'].append(rec)
        metrics['F1'].append(f1)
        metrics['Specificity'].append(spec)
        metrics['AUC'].append(fold_auc)

        fold_metrics_list.append({
            'Fold': fold_idx + 1,
            'Acc': acc, 'Prec': prec, 'Rec': rec, 'F1': f1, 'Spec': spec, 'AUC': fold_auc
        })

        print(f"Fold {fold_idx + 1} -> Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {fold_auc:.4f}")

        #Prepare ROC Data
        fpr, tpr, _ = roc_curve(subject_labels, subject_scores)
        aucs.append(fold_auc)
        # Interpolate to smooth the curve for averaging
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    print("\n" + "-" * 20)
    print("PER-FOLD RESULTS:")
    print("-" * 20)
    df_res = pd.DataFrame(fold_metrics_list)
    if not df_res.empty:
        print(df_res.round(4).to_string(index=False))

    print("\n" + "=" * 40)
    print("🏆 FINAL POOLED RESULTS (Mean ± Std)")
    print("=" * 40)
    for metric, values in metrics.items():
        if values:
            print(f"{metric:<12}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    #CONFUSION MATRIX (Aggregated)
    print("\nGenerating Confusion Matrix...")
    plt.figure(figsize=(6, 5))
    all_preds = []
    all_labels = []

    for fold_idx in range(N_FOLDS):
        file_path = os.path.join(SAVE_DIR, f"preds_fold{fold_idx}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['subject_id'] = df['subject_id'].astype(str)
            scores = df.groupby('subject_id')['prob_pd'].mean()
            labels = df.groupby('subject_id')['label'].first()
            preds = (scores > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels)

    if all_labels:
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Healthy', 'Parkinson'],
                    yticklabels=['Healthy', 'Parkinson'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Aggregated Confusion Matrix')

        # Save Plot
        cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"✅ Confusion Matrix saved to: {cm_path}")
        plt.close()

    #ROC CURVE
    print("Generating ROC Curve...")
    plt.figure(figsize=(8, 6))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='green',
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # Plot Chance Line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # Plot Std Dev (Shading)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # Save Plot
    roc_path = os.path.join(SAVE_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    print(f"✅ ROC Curve saved to: {roc_path}")
    plt.close()

    #REPORT
    report_path = os.path.join(SAVE_DIR, "final_clinical_report.txt")
    with open(report_path, "w") as f:
        f.write("========================================\n")
        f.write("=== FINAL CLINICAL METRICS REPORT ===\n")
        f.write("========================================\n\n")

        f.write("PER-FOLD RESULTS:\n")
        if not df_res.empty:
            f.write(df_res.round(4).to_string(index=False))
        else:
            f.write("No fold data available.")
        f.write("\n\n")

        f.write("========================================\n")
        f.write("🏆 POOLED RESULTS (Mean ± Std)\n")
        f.write("========================================\n")
        for metric, values in metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"{metric:<12}: {mean_val:.4f} ± {std_val:.4f}\n")

    print(f"\n✅ Detailed report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()