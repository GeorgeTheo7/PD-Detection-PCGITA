"""Command-line entry point.

    python main.py prepare  --raw-dir data/raw                  # 1. chunk PC-GITA into 5 s segments
    python main.py extract  --ssl xlsr53                        # 2. cache frozen SSL features (or --ssl all)
    python main.py train    --experiments coattn_xlsr53         # 3. nested speaker-independent 5-fold CV
    python main.py evaluate --experiment coattn_xlsr53          # 4. metrics + plots for one experiment
    python main.py ablations                                    # 5. ablation tables + significance tests

Experiments are defined in src/experiments.py; `--experiments all` runs every variant.
"""
import argparse
import os

from src import config
from src.experiments import EXPERIMENTS


def main():
    p = argparse.ArgumentParser(description="Parkinson's disease detection from speech (PC-GITA)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prepare", help="Select tasks, resample to 16 kHz mono, cut 5 s chunks")
    a.add_argument("--raw-dir", required=True, help="PC-GITA root folder")
    a.add_argument("--out-dir", default="data/processed")
    a.add_argument("--metadata", default=None, help="Optional PC-GITA metadata .xlsx")

    a = sub.add_parser("extract", help="Cache frozen SSL features for all chunks")
    a.add_argument("--processed-dir", default="data/processed")
    a.add_argument("--features-dir", default="data/features")
    a.add_argument("--ssl", nargs="+", default=[config.DEFAULT_SSL], choices=list(config.SSL_MODELS) + ["all"])
    a.add_argument("--batch-size", type=int, default=16)

    a = sub.add_parser("train", help="Nested speaker-independent cross-validation")
    a.add_argument("--processed-dir", default="data/processed")
    a.add_argument("--features-dir", default="data/features")
    a.add_argument("--runs-dir", default="outputs/runs")
    a.add_argument("--folds", default="splits/pcgita_outer_folds.csv",
                   help="Fixed outer-fold assignment (falls back to StratifiedGroupKFold if missing)")
    a.add_argument("--experiments", nargs="+", default=["coattn_xlsr53"],
                   help=f"Experiment names or 'all'. Available: {', '.join(EXPERIMENTS)}")
    a.add_argument("--seeds", type=int, nargs="+", default=config.DEFAULT_SEEDS)
    a.add_argument("--es", default="f1", choices=["f1", "loss"],
                   help="Early-stopping criterion: f1 (main protocol) or loss (ablation)")

    a = sub.add_parser("evaluate", help="Per-fold metrics and plots for one run + summary of all runs")
    a.add_argument("--runs-dir", default="outputs/runs")
    a.add_argument("--experiment", default="coattn_xlsr53")
    a.add_argument("--seed", type=int, default=42)

    a = sub.add_parser("ablations", help="Ablation tables with Wilcoxon signed-rank tests")
    a.add_argument("--runs-dir", default="outputs/runs")

    args = p.parse_args()

    if args.cmd == "prepare":
        from src.preprocess import prepare_pcgita
        prepare_pcgita(args.raw_dir, args.out_dir, args.metadata)

    elif args.cmd == "extract":
        import torch
        from src.features import extract_ssl
        device = "cuda" if torch.cuda.is_available() else "cpu"
        csv = os.path.join(args.processed_dir, "pcgita_chunks.csv")
        chunks = os.path.join(args.processed_dir, "chunks")
        for ssl in (list(config.SSL_MODELS) if "all" in args.ssl else args.ssl):
            extract_ssl(csv, chunks, args.features_dir, ssl, device=device, batch_size=args.batch_size)

    elif args.cmd == "train":
        import numpy as np
        import pandas as pd
        import torch
        from src import experiments
        from src.cv import FeatureStore, run_cv
        from src.features import compute_spec_mfcc, ssl_feature_path

        names = list(EXPERIMENTS) if "all" in args.experiments else args.experiments
        cfgs = {n: experiments.get(n) for n in names}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        csv = os.path.join(args.processed_dir, "pcgita_chunks.csv")
        chunks = os.path.join(args.processed_dir, "chunks")
        df = pd.read_csv(csv)

        ssl = {}
        for name in sorted({c["ssl"] for c in cfgs.values() if c["ssl"]}):
            path = ssl_feature_path(args.features_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found - run: python main.py extract --ssl {name}")
            ssl[name] = torch.from_numpy(np.load(path))
            assert len(ssl[name]) == len(df), f"{path} does not match the chunk CSV - re-run `extract`"
        print("Computing spectrograms and MFCCs...")
        spec, mfcc = compute_spec_mfcc(csv, chunks, device=device)
        store = FeatureStore(df, spec, mfcc, ssl, device)

        for name, cfg in cfgs.items():
            exp_dir = name if args.es == "f1" else f"{name}_esloss"
            for seed in args.seeds:
                run_cv(store, os.path.join(args.runs_dir, exp_dir, f"seed{seed}"), cfg,
                       seed=seed, es_mode=args.es, folds_csv=args.folds, tag=name)

    elif args.cmd == "evaluate":
        from src.evaluate import evaluate_run, summarize_runs
        run = os.path.join(args.runs_dir, args.experiment, f"seed{args.seed}")
        if os.path.isdir(run):
            evaluate_run(run)
        summarize_runs(args.runs_dir)

    elif args.cmd == "ablations":
        from src.evaluate import ablation_report
        ablation_report(args.runs_dir)


if __name__ == "__main__":
    main()
