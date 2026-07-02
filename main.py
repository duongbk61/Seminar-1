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

    # ----- resource-friendly training tricks (see README / config) -----
    ap.add_argument("--small-train", action="store_true",
                    help="preset for limited resources: 2 layers, dropout 0.15, denoising, "
                         "LR scheduler, grad clip, layernorm, 60k train rows — better results on CPU")
    ap.add_argument("--encoder-layers", type=int, default=None,
                    help="number of HGAEConv encoder layers (paper's 124 oversmooths; 2-3 works best)")
    ap.add_argument("--dropout", type=float, default=None, help="dropout rate")
    ap.add_argument("--denoise-std", type=float, default=None,
                    help=">0 adds Gaussian noise to encoder inputs (denoising autoencoder)")
    ap.add_argument("--grad-clip", type=float, default=None, help=">0 clips gradient norm")
    ap.add_argument("--scheduler", dest="scheduler", action="store_true", default=None,
                    help="enable ReduceLROnPlateau on val AUC-PR")
    ap.add_argument("--no-scheduler", dest="scheduler", action="store_false",
                    help="disable the LR scheduler")
    ap.add_argument("--layernorm", dest="layernorm", action="store_true", default=None,
                    help="enable LayerNorm between encoder layers")
    ap.add_argument("--no-layernorm", dest="layernorm", action="store_false",
                    help="disable LayerNorm")
    args = ap.parse_args()

    cfg = Config()
    # 1) preset first, so explicit flags below can still override it
    if args.small_train:
        cfg.encoder_layers = 2
        cfg.dropout = 0.15
        cfg.denoise_std = 0.05
        cfg.use_scheduler = True
        cfg.grad_clip = 5.0
        cfg.use_layernorm = True
        if args.train_size is None and not args.full:
            cfg.train_subsample = 60_000

    # 2) explicit per-knob overrides
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
    if args.encoder_layers is not None:
        cfg.encoder_layers = args.encoder_layers
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.denoise_std is not None:
        cfg.denoise_std = args.denoise_std
    if args.grad_clip is not None:
        cfg.grad_clip = args.grad_clip
    if args.scheduler is not None:
        cfg.use_scheduler = args.scheduler
    if args.layernorm is not None:
        cfg.use_layernorm = args.layernorm
    cfg.device = args.device
    return cfg


if __name__ == "__main__":
    run_training(parse_args())
