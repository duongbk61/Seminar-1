"""Entry point for the one-time training pipeline.

    python main.py                 # train with defaults from src/config.py
    python main.py --epochs 30     # quick run
    python main.py --full          # use the full dataset (no subsampling)

After this finishes, outputs/artifacts.pt exists and you can launch the demo:

    streamlit run app.py
"""
from __future__ import annotations

import argparse

from src.config import Config
from src.train import run_training


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Train the hetero-graph fraud auto-encoder")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-subsample", type=int, default=None)
    ap.add_argument("--test-subsample", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="use the full dataset")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.train_subsample is not None:
        cfg.train_subsample = args.train_subsample
    if args.test_subsample is not None:
        cfg.test_subsample = args.test_subsample
    if args.full:
        cfg.train_subsample = None
        cfg.test_subsample = None
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.device = args.device
    return cfg


if __name__ == "__main__":
    run_training(parse_args())
