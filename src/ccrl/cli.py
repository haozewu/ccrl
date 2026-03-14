from __future__ import annotations

import argparse
import json

from .api import run_ccrl_diag


def build_parser():
    parser = argparse.ArgumentParser(description="Run the distilled CCRL diagnosis pipeline.")
    parser.add_argument("--data", required=True, help="Path to the input pickle file")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated imbalance CV runs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--log-dir", default="logs", help="Directory for tensorboard logs and log files")
    return parser


def main():
    args = build_parser().parse_args()
    result = run_ccrl_diag(
        data_path=args.data,
        repeats=args.repeats,
        seed=args.seed,
        log_dir=args.log_dir,
    )
    print(json.dumps({"mean_f1": result.mean_f1, "std_f1": result.std_f1, "folds": len(result.folds)}, ensure_ascii=False))
