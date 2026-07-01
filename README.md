# Heterogeneous Graph Auto-Encoder for Credit Card Fraud Detection

A **faithful** implementation + interactive demo of:

> S. Majumder, M. T. Singh, R. K. Prasad, et al., *Heterogeneous Graph
> Auto-Encoder for Credit Card Fraud Detection*, International Journal of Computer
> Applications (IJCA), **32(2)**, pp. 124–138, 2025.

Master's seminar project — HUST SoICT, Group 06.

The idea: model transactions as a **heterogeneous graph** (customer ↔ transaction
↔ merchant), encode it with a **custom attention GNN** that implements the paper's
distinctive *intra-edge* attention (Eq. 1–5), add a **variational** latent layer,
and **reconstruct each node type's attributes** with per-type MLP decoders. The
auto-encoder is trained only on **genuine** transactions, so a high reconstruction
error flags an anomaly ⇒ fraud. This sidesteps the class-imbalance problem without
over/under-sampling.

> **This branch is a faithful re-implementation of the paper.** It deliberately
> reverts earlier performance tuning to match the paper's method exactly (no KL
> term, μ+2σ threshold, uniform-neighbour attention, full attribute
> reconstruction). See [Fidelity vs. performance](#fidelity-vs-performance).

## How it works

| Stage | What happens |
|-------|--------------|
| **Graph** | Each transaction row → a `transaction` node + edges to its `customer` and `merchant` nodes. Each node type has its **own** attribute set (customer = demographics, merchant = category + location, transaction = amount/time/distance). |
| **Encoder** | Per-type input projection → stack of custom `HGAEConv` layers. Attention is softmax **across the H heads of each edge**; neighbour messages are aggregated by a **uniform sum** (the paper's ⊕), not neighbour-normalized. |
| **Latent** | Per-type VAE reparameterization (`z = μ + ε·exp(½·logvar)`), applied after each encoder layer. **No KL term** (as in the paper). |
| **Decoder** | Per-type MLP reconstructs the full attribute vector — MSE for continuous features, cross-entropy for categorical — prevalence-weighted across node types. |
| **Detection** | Anomaly score = a transaction's reconstruction error. Verdict = `error ≥ μ + 2σ` of genuine reconstruction errors (paper Eq. 9). |

**Train once / infer many:** training saves everything the demo needs into
`outputs/artifacts.pt`; the demo loads it and scores any *single* new transaction
instantly on CPU by building a tiny 3-node graph. New customers/merchants are
handled via cold-start (featurized from the incoming record). **No retraining
needed to demo.**

## Project layout

```
code/
├─ data/                  # put fraudTrain.csv / fraudTest.csv here (see data/README.md)
├─ src/
│  ├─ config.py           # all hyper-parameters (paper Table 3)
│  ├─ data_prep.py        # per-type featurizers + hetero-graph construction
│  ├─ hgae_conv.py        # custom intra-edge attention layer (paper Eq. 1–5)
│  ├─ model.py            # encoder + VAE latent + per-type decoders + loss
│  ├─ train.py            # train-once pipeline, saves artifacts + plots
│  ├─ evaluate.py         # μ+2σ and F1-sweep thresholds, metrics, Figure-5 plots
│  ├─ scorer.py           # FraudScorer: real-time single-transaction scoring
│  └─ utils.py            # seeding + device selection
├─ tests/                 # pytest suite (17 tests)
├─ main.py                # `python main.py` → train
├─ app.py                 # `streamlit run app.py` → demo
└─ requirements.txt
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/CMD)
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

This installs `torch`, `torch_geometric`, `pandas`, `numpy`, `scikit-learn`,
`matplotlib`, `streamlit`, and `pytest`. CPU-only is fine; a CUDA GPU is used
automatically when available.

## How to run — quick start

First get the dataset (one time): download the Kaggle
`kartik2112/fraud-detection` dataset and place `fraudTrain.csv` and
`fraudTest.csv` in `data/` (or `archive/`) — see [`data/README.md`](data/README.md).
Then:

```bash
python main.py --epochs 40 --train-size 50000   # train, evaluate, save artifacts + plots
streamlit run app.py                            # open the interactive demo in the browser
```

Train/test sizes are drawn from the real data via `--train-size` / `--test-size`
(see [How to train](#how-to-train)); omit them for the config defaults, or pass
`--full` to use the entire dataset.

In the demo you can load a random genuine/fraud example, pick a known
customer/merchant or invent brand-new ones (cold-start), tweak the amount / hour /
category, and watch the reconstruction error cross the fraud threshold — with a
per-feature breakdown of *why*. A **📈 Training progress** panel plots the
per-epoch loss and validation AUC-PR curves saved with the model.

## How to train

`python main.py` runs the full train-once pipeline: load data → fit per-type
featurizers on **genuine** rows only → build the graph → train the auto-encoder →
pick thresholds → evaluate on the test graph → save artifacts, metrics, and plots.

```bash
python main.py                       # defaults from src/config.py
python main.py --epochs 40           # fewer epochs (quick run)
python main.py --train-size 20000    # draw 20k rows from the real train file (CPU-friendly)
python main.py --test-size 20000     # draw 20k rows from the real test file
python main.py --full                # use the ENTIRE real dataset (no sampling)
python main.py --device cpu          # force CPU (also: cuda | auto)
python main.py --seed 123            # change the RNG seed
```

CLI flags (see `main.py`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--epochs N` | 150 | training epochs (early-stops on val AUC-PR) |
| `--train-size N` | 120000 | rows drawn from the real train file (all fraud always kept) |
| `--test-size N` | full | rows drawn from the real test file (omit = full test set) |
| `--full` | off | use the entire real dataset (ignores `--train-size`/`--test-size`) |
| `--device` | auto | `auto` \| `cpu` \| `cuda` |
| `--seed N` | 42 | random seed |

`--train-size`/`--test-size` are always sampled from the **real** dataset; all
fraud rows are kept (the auto-encoder trains on genuine rows, and fraud is used
for validation/threshold selection). All hyper-parameters (hidden size 64, 16
heads, dropout 0.4, weight-decay 0.01, etc.) live in
[`src/config.py`](src/config.py) and follow the paper's **Table 3**.

> **Dataset required.** There is no synthetic fallback — training aborts with a
> clear error if `fraudTrain.csv` / `fraudTest.csv` are not found in `data/` or
> `archive/`.

## How to test

The `tests/` directory holds a pytest suite (17 tests) covering the featurizers,
graph construction, the custom attention layer (attention-sums-to-1 across heads,
uniform-sum aggregation), the model/loss (no-KL, MSE+CE), the μ+2σ threshold, an
in-memory training smoke test, and the scorer (load + cold-start).

```bash
python -m pytest                       # run the whole suite (from the repo root)
python -m pytest -v                    # verbose, one line per test
python -m pytest -q                    # quiet summary
python -m pytest tests/test_model.py   # a single file
python -m pytest tests/test_hgae_conv.py::test_uniform_neighbour_aggregation_is_a_sum
```

The tests need only `torch` / `torch_geometric` / `pytest` — no dataset or trained
model. Run them from the project root so `src/` imports resolve.

## Outputs

After training you get:

* `outputs/artifacts.pt` — everything the demo needs (weights, per-type
  featurizers, reference profiles, both thresholds, metrics).
* `outputs/metrics.json` — test metrics at the μ+2σ operating point **and** at the
  F1-sweep operating point.
* `outputs/plots/*.png` — training loss, reconstruction-error distribution,
  F1-vs-threshold, ROC and Precision-Recall curves (paper Figure 5).

## Fidelity vs. performance

Because this build is faithful to the paper, expect **modest detection quality**
on the highly imbalanced test set (the paper's own ROC-AUC of 0.85 is below its
GNN baselines). The main reason is architectural: the paper omits the KL
divergence term, so the variational latent injects noise that the training signal
cannot fully overcome. This is a deliberate property of the paper's design, not a
bug — reintroducing a small KL term (a *deviation* from the paper) is the known
remedy and is what a performance-tuned variant would do.

## Deviations from the paper

Only **two**, both because the paper is unrunnable as literally written (each is
documented in [`src/config.py`](src/config.py)):

* **Encoder depth.** Table 3 lists "encoder layers = 124"; 124 message-passing
  layers oversmooth catastrophically, so `encoder_layers` defaults to **2**.
* **Reparameterization.** The literal formula takes `log()` of raw activations
  (produces NaNs), so a standard `μ`/`logvar` parameterization is used instead.

Everything else matches Table 3: 16 heads, dropout 0.4, regularization 0.01, **no
KL term**, and **no latent bottleneck** (latent dim = hidden = 64).
