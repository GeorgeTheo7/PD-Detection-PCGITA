"""Command-line entry point.

    python main.py prepare  --raw-dir data/raw          # 1. chunk PC-GITA into 5 s segments
    python main.py extract                              # 2. cache frozen Wav2Vec2 features
    python main.py train    --seeds 42 1 2 3 4          # 3. speaker-independent nested 5-fold CV
    python main.py evaluate                             # 4. metrics, plots, multi-seed summary
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser(description="Parkinson's disease detection from speech (PC-GITA)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prepare", help="Select tasks, resample to 16 kHz mono, cut 5 s chunks")
    a.add_argument("--raw-dir", required=True, help="PC-GITA root folder")
    a.add_argument("--out-dir", default="data/processed")
    a.add_argument("--metadata", default=None, help="Optional PC-GITA metadata .xlsx")

    a = sub.add_parser("extract", help="Cache frozen Wav2Vec2 features for all chunks")
    a.add_argument("--processed-dir", default="data/processed")
    a.add_argument("--features", default="data/features/w2v_frozen_fp16.npy")
    a.add_argument("--batch-size", type=int, default=16)

    a = sub.add_parser("train", help="Nested speaker-independent cross-validation")
    a.add_argument("--processed-dir", default="data/processed")
    a.add_argument("--features", default="data/features/w2v_frozen_fp16.npy")
    a.add_argument("--runs-dir", default="outputs/runs")
    a.add_argument("--folds", default="splits/pcgita_outer_folds.csv",
                   help="Fixed outer-fold assignment (falls back to StratifiedGroupKFold if missing)")
    a.add_argument("--seeds", type=int, nargs="+", default=[42])
    a.add_argument("--es", nargs="+", default=["f1"], choices=["f1", "loss"],
                   help="Early-stopping criterion(s): f1 (thesis protocol) and/or loss (ablation)")

    a = sub.add_parser("evaluate", help="Per-fold metrics, plots and multi-seed summary")
    a.add_argument("--runs-dir", default="outputs/runs")
    a.add_argument("--run", default="es_f1_seed42", help="Run to plot in detail")

    args = p.parse_args()

    if args.cmd == "prepare":
        from src.preprocess import prepare_pcgita
        prepare_pcgita(args.raw_dir, args.out_dir, args.metadata)

    elif args.cmd in ("extract", "train"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        csv = os.path.join(args.processed_dir, "pcgita_chunks.csv")
        chunks = os.path.join(args.processed_dir, "chunks")

        if args.cmd == "extract":
            from src.features import extract_wav2vec2
            extract_wav2vec2(csv, chunks, args.features, device=device, batch_size=args.batch_size)
        else:
            import numpy as np
            import pandas as pd
            from src.cv import FeatureStore, run_cv
            from src.features import compute_spec_mfcc

            df = pd.read_csv(csv)
            w2v = torch.from_numpy(np.load(args.features))
            assert len(w2v) == len(df), "Feature file does not match the chunk CSV - re-run `extract`"
            print("Computing spectrograms and MFCCs...")
            spec, mfcc = compute_spec_mfcc(csv, chunks, device=device)
            store = FeatureStore(df, spec, mfcc, w2v, device)
            for mode in args.es:
                for seed in args.seeds:
                    run_cv(store, os.path.join(args.runs_dir, f"es_{mode}_seed{seed}"),
                           seed=seed, es_mode=mode, folds_csv=args.folds)

    elif args.cmd == "evaluate":
        from src.evaluate import evaluate_run, summarize_runs
        run = os.path.join(args.runs_dir, args.run)
        if os.path.isdir(run):
            evaluate_run(run)
        summarize_runs(args.runs_dir)


if __name__ == "__main__":
    main()
