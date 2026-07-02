"""Central configuration for the Heterogeneous Graph Auto-Encoder fraud detector.

Hyper-parameters follow the paper's **Table 3** ("Values for different parameters
used in the model") of "Heterogeneous Graph Auto-Encoder for Credit Card Fraud
Detection" (Majumder et al., IJCA 32(2) 2025). (Table 4 is the results table.)

This is the FAITHFUL implementation. Only two values are not used verbatim,
because the paper is unrunnable as written:
  * "Number of Layers for the Encoder (l) = 124" -> `encoder_layers = 2`.
    124 message-passing layers oversmooth catastrophically (embeddings collapse).
  * The literal reparameterization `mean(h) + eps*exp(0.5*log(h))` takes log() of
    raw activations (NaN); the model uses standard mu/logvar heads instead.
Everything else matches the paper: 16 heads, dropout 0.4, regularization 0.01,
NO KL term, NO latent bottleneck (latent = hidden).
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

    # ----- how many rows to draw from the REAL dataset (override via main.py) -----
    # Rows kept from the TRAIN file (CPU-friendly). The auto-encoder only learns
    # from genuine rows, so we keep every fraud row (they go to validation for
    # threshold selection) plus a genuine pool. None = the full file. CLI: --train-size.
    train_subsample: int | None = 120_000
    keep_all_fraud: bool = True
    # TEST is scored in a single cheap forward pass, so evaluate on the FULL test
    # set to keep the natural class imbalance (faithful metrics, like the paper).
    # None = the full file. CLI: --test-size.
    test_subsample: int | None = None

    # ----- model (paper Table 3) -----
    hidden_dim: int = 64          # "Size of Hidden Layers" = 64
    heads: int = 16               # "Number of heads (H)" = 16
    encoder_layers: int = 2       # FORCED DEVIATION: Table 3 says 124 -> oversmooths
    decoder_hidden: int = 32      # "Number of Layers for the Decoder" = 64 (width)
    latent_dim: int = 64          # = hidden_dim; the paper has NO bottleneck
    dropout: float = 0.4          # "Dropout Rate" = 0.4

    # ----- training tricks (opt-in; off = paper-faithful; see `--small-train`) -----
    denoise_std: float = 0.0      # >0: add Gaussian noise to encoder inputs, reconstruct clean
    use_scheduler: bool = False   # ReduceLROnPlateau on val AUC-PR
    grad_clip: float = 0.0        # >0: clip gradient norm
    use_layernorm: bool = False   # LayerNorm between HGAEConv layers (fights oversmoothing)

    # ----- optimisation -----
    lr: float = 2e-3
    weight_decay: float = 0.01    # "Regularization Rate" = 0.01
    epochs: int = 150
    beta: float = 5e-4             # the paper has NO KL term (reconstruction-only)
    val_fraction: float = 0.15
    early_stop_patience: int = 25

    seed: int = 42
    device: str = "cpu"           # no CUDA GPU detected on this machine

    # ----- demo: 3D graph visualization -----
    # Max transaction nodes sampled into the demo's interactive 3D graph (their
    # customers/merchants are added on top). Fraud is over-sampled to ~viz_fraud_frac
    # of the transactions so the genuine/fraud contrast is visible.
    viz_max_nodes: int = 200
    viz_fraud_frac: float = 0.2

    output_dir: Path = OUTPUT_DIR

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        assert self.hidden_dim % self.heads == 0, "hidden_dim must be divisible by heads"


def get_config() -> "Config":
    return Config()
