# Parkinson's Disease Detection from Speech (PC-GITA)

A multi-branch deep learning model that detects Parkinson's disease (PD) from connected Spanish speech.
It combines three views of the same 5-second speech segment: a **spectrogram image** (ResNet-18), an
**MFCC sequence** (BiLSTM) and **self-supervised Wav2Vec 2.0 embeddings**. A **co-attention** mechanism
uses the first two to decide which Wav2Vec2 frames matter most.

This work is based on my diploma thesis *"Parkinson's Disease Detection Using Speech"*
(National Technical University of Athens, 2026). It is evaluated with a strict
**speaker-independent** protocol, so no speaker ever appears in both training and test data.

<p align="center"><img src="docs/architecture.png" width="520" alt="Model architecture"></p>

## Results

Subject-level results with nested 5-fold speaker-independent cross-validation on PC-GITA
(50 PD + 50 healthy speakers, Read Text + Monologue). The model was trained with **5 random seeds**,
using the same outer test folds each time. Each seed's metrics are the mean over its 5 folds. The table shows
**mean ± std across the 5 seeds**.

| Early stopping | Accuracy | Recall (sensitivity) | Specificity | F1 | AUC |
|---|---|---|---|---|---|
| **Subject-level F1 (main protocol)** | **86.4% ± 2.4** | **80.8% ± 5.2** | **92.0% ± 4.0** | **84.5% ± 3.5** | **0.962 ± 0.009** |
| Validation loss (ablation) | 83.8% ± 2.7 | 74.8% ± 5.4 | 92.8% ± 3.6 | 80.3% ± 4.2 | 0.959 ± 0.014 |

Per-seed numbers: [`results/multiseed_summary.csv`](results/multiseed_summary.csv).

**Which speech task helps more?** These numbers come from the seed-42 run. The same models were evaluated
on only one task's segments per speaker:

| Speech used at test time | Accuracy | AUC |
|---|---|---|
| Read Text only | 77.0% ± 12.9 | 0.900 ± 0.049 |
| Monologue only | 84.0% ± 11.6 | 0.946 ± 0.052 |
| **Both (default)** | **84.0% ± 10.2** | **0.962 ± 0.035** |

Spontaneous speech (monologue) is more informative than reading, and combining both tasks gives the best ranking (AUC).

### Main run (seed 42)

| Fold | Accuracy | Precision | Recall | Specificity | F1 | AUC |
|---|---|---|---|---|---|---|
| 1 | 85.0% | 81.8% | 90.0% | 80.0% | 85.7% | 0.920 |
| 2 | 95.0% | 100.0% | 90.0% | 100.0% | 94.7% | 1.000 |
| 3 | 75.0% | 100.0% | 50.0% | 100.0% | 66.7% | 0.980 |
| 4 | 70.0% | 100.0% | 40.0% | 100.0% | 57.1% | 0.920 |
| 5 | 95.0% | 100.0% | 90.0% | 100.0% | 94.7% | 0.990 |
| **Mean ± std** | **84.0% ± 10.2** | **96.4% ± 7.3** | **72.0% ± 22.3** | **96.0% ± 8.0** | **79.8% ± 15.3** | **0.962 ± 0.035** |

<p align="center">
  <img src="results/main_run_seed42/confusion_matrix.png" width="340" alt="Confusion matrix">
  <img src="results/main_run_seed42/roc_curve.png" width="380" alt="ROC curve">
</p>

**Takeaways and limitations**
- The model separates PD from healthy speakers well (AUC ≈ 0.96). With the default 0.5 threshold it is
  **conservative**: specificity is higher than sensitivity, and most errors are PD speakers classified as healthy.
- With 20 test speakers per fold, one speaker equals 5% accuracy, so per-fold variance is high. Some PD speakers
  (mostly in fold 4) are missed in almost every run.
- PC-GITA is small (100 speakers, one language, one recording setup). Generalization to other corpora is not shown here.

## Method

**Preprocessing.** Read Text (*"ayer fui al médico…"*) and Monologue recordings, unnormalized version.
Each recording is converted to mono, resampled to 16 kHz and cut into non-overlapping 5 s chunks. The last chunk
is zero-padded, or dropped if it is shorter than 2.5 s. This gives 1,306 chunks from 100 speakers.

**Three branches per chunk**
| Branch | Input | Encoder | Output |
|---|---|---|---|
| Spectrogram | log-Mel (200 bands, 40 ms window, 10 ms hop) + Δ + ΔΔ as a 3-channel 224×224 image, ImageNet-normalized | ResNet-18 (ImageNet-pretrained) | 512-d |
| MFCC | 40 MFCCs × 501 frames | BiLSTM (64 units per direction) + linear | 128-d |
| Raw audio | 80,000-sample waveform | Wav2Vec2 XLSR-53 (Spanish), **frozen** | 249 × 1024 frames |

**Co-attention fusion.** The spectrogram and MFCC vectors form a 640-d *global context*, which is projected to a
128-d query. Each Wav2Vec2 frame is projected to a 128-d key. Dot-product attention with softmax weights the
frames, and their weighted sum (1024-d) is concatenated with the global context (1664-d). An MLP
(1664 → 64 → 2, dropout 0.2) produces the chunk-level prediction.

**Evaluation protocol**
- **Outer loop:** 5 folds, stratified and grouped by speaker. Every speaker (both tasks) is in exactly one test
  fold. The fold assignment is fixed in [`splits/pcgita_outer_folds.csv`](splits/pcgita_outer_folds.csv).
- **Inner split:** 15% of the training speakers are held out *only* for early stopping (patience 3, max 15
  epochs). The outer test fold is used once, at the end.
- **Subject-level diagnosis:** chunk probabilities are averaged per speaker and thresholded at 0.5.
- Adam (lr 1e-4), batch size 64, cross-entropy loss, gradient clipping at 1.0. Fully seeded: re-running seed 42
  on the same chunk files reproduced the main run exactly. Regenerating the chunks from scratch can change the row
  order, and therefore the batch order, so expect small differences in that case.

## Ablation studies

Every variant uses the same folds, seeds and training protocol as the main model; the variants are defined in
[`src/experiments.py`](src/experiments.py). The reference for the ablations is the co-attention model with a
frozen **HuBERT Base** backbone. Values are **mean ± std across 5 seeds** (each seed is the mean over 5 folds),
in %. *p* is a Wilcoxon signed-rank test against the reference, pairing the 25 seed×fold results. This is valid
because every variant is tested on exactly the same speakers.

**SSL backbone** (co-attention, all three branches)

| Model | Precision | Recall | F1 | Accuracy | Specificity | AUC | p (Acc) | p (AUC) |
|---|---|---|---|---|---|---|---|---|
| wav2vec2 Base | 92.23 ± 3.16 | 80.40 ± 4.98 | 83.72 ± 2.52 | 85.80 ± 1.79 | 91.20 ± 3.63 | 96.32 ± 0.77 | 0.434 | 0.400 |
| wav2vec2 XLSR-53 | 92.94 ± 3.43 | 80.80 ± 5.22 | 84.46 ± 3.51 | 86.40 ± 2.41 | 92.00 ± 4.00 | 96.16 ± 0.86 | 0.838 | 0.765 |
| **HuBERT Base** | 92.33 ± 3.82 | 81.60 ± 4.77 | 84.55 ± 2.51 | 86.40 ± 1.82 | 91.20 ± 4.60 | 96.52 ± 0.76 | – | – |

**Fusion method** (HuBERT Base, all three branches)

| Model | Precision | Recall | F1 | Accuracy | Specificity | AUC | p (Acc) | p (AUC) |
|---|---|---|---|---|---|---|---|---|
| Concatenation | 92.18 ± 2.33 | 79.20 ± 4.38 | 82.54 ± 4.33 | 85.00 ± 2.74 | 90.80 ± 3.03 | 95.96 ± 1.37 | 0.339 | 0.598 |
| MLB | 91.66 ± 3.67 | 80.40 ± 3.85 | 83.20 ± 3.00 | 85.40 ± 2.61 | 90.40 ± 4.56 | 95.60 ± 0.71 | 0.465 | 0.093 |
| MFB | 91.63 ± 1.87 | 78.40 ± 2.61 | 82.02 ± 2.40 | 84.60 ± 2.30 | 90.80 ± 2.68 | 96.04 ± 1.05 | 0.158 | 0.838 |
| **Co-attention** | 92.33 ± 3.82 | 81.60 ± 4.77 | 84.55 ± 2.51 | 86.40 ± 1.82 | 91.20 ± 4.60 | 96.52 ± 0.76 | – | – |

**Input modalities** (HuBERT Base)

| Model | Precision | Recall | F1 | Accuracy | Specificity | AUC | p (Acc) | p (AUC) |
|---|---|---|---|---|---|---|---|---|
| HuBERT only | 72.56 ± 4.86 | 64.00 ± 9.06 | 63.14 ± 6.08 | 66.80 ± 5.26 | 69.60 ± 12.76 | 77.08 ± 3.92 | <0.001 | <0.001 |
| MFCC + HuBERT | 73.88 ± 6.41 | 79.60 ± 6.84 | 75.12 ± 5.30 | 73.80 ± 5.97 | 68.00 ± 13.19 | 85.84 ± 2.53 | <0.001 | <0.001 |
| Spectrogram + MFCC (no SSL) | 91.57 ± 3.06 | 77.60 ± 3.29 | 81.53 ± 2.88 | 83.80 ± 2.68 | 90.00 ± 3.74 | 96.84 ± 0.55 | 0.029 | 0.596 |
| Spectrogram + HuBERT | 91.59 ± 3.23 | 81.60 ± 3.85 | 84.30 ± 3.77 | 85.60 ± 3.21 | 89.60 ± 4.34 | 96.32 ± 1.35 | 0.626 | 0.766 |
| **All three (co-attention)** | 92.33 ± 3.82 | 81.60 ± 4.77 | 84.55 ± 2.51 | 86.40 ± 1.82 | 91.20 ± 4.60 | 96.52 ± 0.76 | – | – |

**Takeaways**
- **Backbone:** the three frozen SSL models perform almost identically, with no significant differences.
  HuBERT Base gives the highest AUC and the lowest variance across seeds.
- **Fusion:** co-attention has the best F1, accuracy, recall and AUC, but its margins over concatenation, MLB and
  MFB (about 1–2 points) are within the seed-to-seed variation and not statistically significant.
- **Modalities:** the **spectrogram branch is essential**. Every configuration without it drops to 67–74%
  accuracy. Mean-pooled frozen HuBERT on its own is weak (66.8%). Adding the SSL branch to spectrogram + MFCC
  raises accuracy (83.8% → 86.4%, *p* = 0.029) and recall, while AUC stays similar.

Run all variants with `python main.py train --experiments all`, and build these tables with
`python main.py ablations`.

## Repository structure

```
├── main.py                    # CLI: prepare | extract | train | evaluate | ablations
├── src/
│   ├── config.py              # audio, feature and training settings
│   ├── preprocess.py          # task selection, resampling, 5 s chunking, speaker IDs
│   ├── features.py            # SSL feature caching, spectrogram and MFCC computation
│   ├── models.py              # ResNet-18 / BiLSTM encoders, co-attention and ablation fusions
│   ├── experiments.py         # main model + ablation variants
│   ├── cv.py                  # nested speaker-independent cross-validation
│   └── evaluate.py            # subject-level metrics, multi-seed summary, ablation tables, plots
├── splits/pcgita_outer_folds.csv
├── notebooks/run_on_colab.ipynb
├── results/                   # metrics and figures reported above
└── docs/architecture.png
```

## Reproducing the results

### 1. Get the data
PC-GITA is not included and cannot be redistributed. It is available on request from the GITA Lab,
Universidad de Antioquia (Orozco-Arroyave et al., 2014). Keep the original folder layout:
```
<PCGITA_ROOT>/read text/ayerfuialmedico/sin normalizar/{hc,pd}/*.wav
<PCGITA_ROOT>/monologue/sin normalizar/{hc,pd}/*.wav
```

### 2. Run
```bash
pip install -r requirements.txt

python main.py prepare  --raw-dir <PCGITA_ROOT> [--metadata <PCGITA_metadata.xlsx>]
python main.py extract  --ssl xlsr53                          # frozen SSL features (~1 min on a T4)
python main.py train    --experiments coattn_xlsr53           # 5 seeds by default
python main.py evaluate --experiment coattn_xlsr53
```
Outputs go to `outputs/runs/<experiment>/seed<seed>/`: per-fold predictions, logs, metrics and plots.
Training is resumable, so finished folds are skipped. Add `--es loss` to `train` for the early-stopping ablation.

To run all ablation variants and build the ablation tables:
```bash
python main.py extract   --ssl all
python main.py train     --experiments all
python main.py ablations                                       # tables + Wilcoxon tests
```

The whole pipeline runs on a free Colab T4. See
[`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb).

## References
- J. R. Orozco-Arroyave, J. D. Arias-Londoño, J. F. Vargas-Bonilla, M. C. González-Rátiva, E. Nöth.
  *New Spanish speech corpus database for the analysis of people suffering from Parkinson's disease.* LREC 2014.
- Y. Shao, T. Fang. *Alzheimer's Disease Detection Using Co-Attention Mechanism for Acoustic and ASR-Transcribed
  Text Features*, 2025. This was the inspiration for the co-attention fusion.
- A. Conneau et al. *Unsupervised Cross-lingual Representation Learning for Speech Recognition* (XLSR-53), 2020.

## License
Code: MIT (see [LICENSE](LICENSE)). PC-GITA data is subject to its own license and is not distributed here.
