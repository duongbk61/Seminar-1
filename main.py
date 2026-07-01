"""Entry point for the one-time training pipeline.

    python main.py                        # train with defaults from src/config.py
    python main.py --epochs 30            # quick run
    python main.py --train-size 50000     # sample 50k rows from the real train file
    python main.py --test-size 20000      # sample 20k rows from the real test file
    python main.py --full                 # use the entire real dataset (no sampling)

The dataset is the real Kaggle 'kartik2112/fraud-detection' data; place
fraudTrain.csv & fraudTest.csv in data/ or archive/. Train/test sizes are drawn
from that real data via the flags below (all fraud rows are always kept).

After this finishes, outputs/artifacts.pt exists and you can launch the demo:

    streamlit run app.py
"""
from __future__ import annotations

import argparse

from src.config import Config
from src.train import run_training


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Train the hetero-graph fraud auto-encoder")
    ap.add_argument("--epochs", type=int, default=None, help="training epochs")
    ap.add_argument("--train-size", type=int, default=None,
                    help="number of rows to draw from the real train file (all fraud kept); "
                         "omit for the config default")
    ap.add_argument("--test-size", type=int, default=None,
                    help="number of rows to draw from the real test file; omit for the full test set")
    ap.add_argument("--full", action="store_true",
                    help="use the entire real dataset (ignore --train-size/--test-size)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.train_size is not None:
        cfg.train_subsample = args.train_size
    if args.test_size is not None:
        cfg.test_subsample = args.test_size
    if args.full:
        cfg.train_subsample = None
        cfg.test_subsample = None
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.device = args.device
    return cfg


if __name__ == "__main__":
    run_training(parse_args())
