"""Central configuration for the Heterogeneous Graph Auto-Encoder fraud detector.

The hyper-parameters follow Table 4 of the paper
"Heterogeneous Graph Auto-Encoder for Credit Card Fraud Detection"
(Singh et al., arXiv:2410.08121) where they are sensible.

Two values reported in Table 4 are almost certainly typos and are NOT used
verbatim:
  * "Number of Layers for the Encoder (l) = 124"  -> 124 message-passing layers
    would over-smooth a GNN catastrophically. We expose `encoder_layers`
    (default 2) instead.
  * "Number of Layers for the Decoder = 64" -> interpreted as the decoder
    hidden width (`decoder_hidden = 64`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class Config:
    # ----- data files (Kaggle kartik2112/fraud-detection schema) -----
    train_csv: Path = DATA_DIR / "fraudTrain.csv"
    test_csv: Path = DATA_DIR / "fraudTest.csv"

    # ----- subsampling (CPU-friendly, keeps class imbalance) -----
    # Transactions kept from the TRAIN file (CPU-friendly). The auto-encoder only
    # learns from genuine rows, so we keep every fraud row (they go to validation
    # for threshold selection) plus a large genuine pool. Set None for the full file.
    train_subsample: int | None = 120_000
    keep_all_fraud: bool = True
    # TEST is scored in a single cheap forward pass, so evaluate on the FULL test
    # set to keep the natural class imbalance (faithful metrics, like the paper).
    test_subsample: int | None = None

    # ----- model (Table 4) -----
    hidden_dim: int = 64          # "Size of Hidden Layers" = 64
    heads: int = 8                # paper says 16; 8 is faster/stabler on CPU (hidden % heads == 0)
    encoder_layers: int = 2       # paper's "124" is implausible; 2 HGT layers
    decoder_hidden: int = 64      # paper's "Decoder = 64"
    # For reconstruction-based anomaly detection the latent MUST be a bottleneck
    # (latent_dim < n_features ~= 23) so the AE cannot learn the identity map and
    # genuine vs. fraud reconstruction error separates. Paper's 0.4 dropout caused
    # severe under-fitting here, so we use a light value (genuine must reconstruct
    # well for the anomaly signal to emerge).
    latent_dim: int = 12
    dropout: float = 0.1

    # ----- optimisation -----
    lr: float = 2e-3
    weight_decay: float = 1e-5    # paper's 0.01 over-regularised; AE needs to fit genuine
    epochs: int = 150
    beta: float = 5e-4            # KL weight in the VAE objective (kept small -> near-AE)
    val_fraction: float = 0.15    # genuine rows held out from training for thresholding
    early_stop_patience: int = 25

    seed: int = 42
    device: str = "cpu"           # no CUDA GPU detected on this machine

    output_dir: Path = OUTPUT_DIR

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        assert self.hidden_dim % self.heads == 0, "hidden_dim must be divisible by heads"


def get_config() -> "Config":
    return Config()
