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
├─ data/                  # put fraudTrain.csv / fraudTest.csv here (gitignored)
├─ docs/                  # implementation & preprocessing notes
├─ src/
│  ├─ config.py           # all hyper-parameters (paper Table 3)
│  ├─ data_prep.py        # per-type featurizers + hetero-graph construction
│  ├─ hgae_conv.py        # custom intra-edge attention layer (paper Eq. 1–5)
│  ├─ model.py            # encoder + VAE latent + per-type decoders + loss
│  ├─ train.py            # train-once pipeline, saves artifacts + plots
│  ├─ evaluate.py         # μ+2σ and F1-sweep thresholds, metrics, Figure-5 plots
│  ├─ scorer.py           # FraudScorer: real-time single-transaction scoring
│  ├─ graph_viz.py        # self-contained 3D force-graph HTML for the demo
│  └─ utils.py            # seeding + device selection
├─ tests/                 # pytest suite (33 tests)
├─ main.py                # `python main.py` → train
├─ app.py                 # `python -m streamlit run app.py` → demo
├─ outputs/               # artifacts + metrics + plots (created by training, gitignored)
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

First get the dataset (one time): download the
[Kaggle `kartik2112/fraud-detection` dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
and place `fraudTrain.csv` (~350 MB) and `fraudTest.csv` (~150 MB) in `data/`
(or `archive/`). Both locations are gitignored — never commit the CSVs. Then:

```bash
python main.py --epochs 40 --train-size 50000   # train, evaluate, save artifacts + plots
python -m streamlit run app.py                  # open the interactive demo in the browser
```

Train/test sizes are drawn from the real data via `--train-size` / `--test-size`
(see [How to train](#how-to-train)); omit them for the config defaults, or pass
`--full` to use the entire dataset. See [The demo app](#the-demo-app) for what the
demo contains and how to use it.

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
| `--train-size N` | 200000 | rows drawn from the real train file (all fraud always kept) |
| `--test-size N` | 100000 | rows drawn from the real test file |
| `--full` | off | use the entire real dataset (ignores `--train-size`/`--test-size`) |
| `--device` | auto | `auto` \| `cpu` \| `cuda` |
| `--seed N` | 42 | random seed |

There are also **opt-in training tricks** — all *off* by default so the default
run stays paper-faithful: `--small-train` (a preset for limited hardware: 2
layers, dropout 0.15, denoising, LR scheduler, gradient clipping, LayerNorm, 60k
train rows), plus the individual knobs `--encoder-layers`, `--dropout`,
`--denoise-std`, `--grad-clip`, `--scheduler`/`--no-scheduler`, and
`--layernorm`/`--no-layernorm`.

`--train-size`/`--test-size` are always sampled from the **real** dataset; all
fraud rows are kept (the auto-encoder trains on genuine rows, and fraud is used
for validation/threshold selection). All hyper-parameters (hidden size 64, 16
heads, dropout 0.4, weight-decay 0.01, etc.) live in
[`src/config.py`](src/config.py) and follow the paper's **Table 3**.

> **Dataset required.** There is no synthetic fallback — training aborts with a
> clear error if `fraudTrain.csv` / `fraudTest.csv` are not found in `data/` or
> `archive/`.

## How to test

The `tests/` directory holds a pytest suite (33 tests) covering the featurizers,
graph construction, the custom attention layer (attention-sums-to-1 across heads,
uniform-sum aggregation), the model/loss (no-KL, MSE+CE), the μ+2σ threshold, an
in-memory training smoke test, the scorer (load + cold-start), and the 3D graph
visualization helpers.

```bash
python -m pytest                       # run the whole suite (from the repo root)
python -m pytest -v                    # verbose, one line per test
python -m pytest -q                    # quiet summary
python -m pytest tests/test_model.py   # a single file
python -m pytest tests/test_hgae_conv.py::test_uniform_neighbour_aggregation_is_a_sum
```

The tests need only `torch` / `torch_geometric` / `pytest` — no dataset or trained
model. Run them from the project root so `src/` imports resolve.

## The demo app

An interactive Streamlit app that scores a **single** transaction in real time
(CPU, no retraining). It loads `outputs/artifacts.pt`, so **train first**
([How to train](#how-to-train)).

### Launch

```bash
python -m streamlit run app.py
```

This opens `http://localhost:8501` in your browser (Ctrl+C in the terminal to
stop). Use the `python -m streamlit` form — the bare `streamlit` command only
works if Python's `Scripts/` directory is on your PATH (often not the case on
Windows). If the app shows *"No trained model found at outputs/artifacts.pt"*, run
`python main.py` first.

### What's in it

**Sidebar — model card:** ROC-AUC / AUC-PR / F1 / Precision / Recall on the test
set at the μ+2σ threshold (plus the F1-sweep F1 for comparison), the decision
threshold, and how many known customers/merchants have stored profiles.

**🛰️ Live fraud monitoring (expander):** a SOC-style live stream that replays
test transactions through the scorer — verdict feed, running fraud-rate metrics,
and a geographic arc map (cardholder home → merchant) with an optional
fraud-density heatmap.

**📊 Data characteristics (expander):** class-balance pies for train/test and
fraud-rate breakdowns by amount, category, hour, age, and cardholder–merchant
distance.

**🌐 Transaction graph — 3D (expander):** an interactive 3D force-graph of a
sampled customer ↔ transaction ↔ merchant subgraph; transactions you score in
the demo are injected as new nodes.

**1 · Pick a transaction:**

* Quick-load buttons: 🎲 *Random genuine*, 🚨 *Random fraud* (real examples from
  the test set), 🆕 *New customer*, 🆕 *New merchant* (invent an unseen entity →
  cold-start).
* Three editable panels — **👤 Customer** (known cardholder → stored genuine
  profile, or new → cold-start), **🏪 Merchant** (known or new), and **💳
  Transaction** (amount, category, hour of day, date).

**2 · Score it:** the **🔎 Check transaction** button runs the model and shows the
verdict (✅ NON-FRAUD / 🚨 FRAUD), the reconstruction error vs. threshold and their
ratio, an anomaly-level bar (1.0 = at threshold), known/cold-start tags, a
**top contributing features** table explaining *why* it scored that way, and an
**anomaly fingerprint** radar (tight = genuine, spiky = fraud) — plus the raw
record in an expander.

**Try this:** load a random genuine transaction (scores ✅), then push the amount
up, set the hour to 3 AM, or switch to a brand-new customer/merchant, and re-score
— watch the reconstruction error climb past the threshold and flip to 🚨.

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

Each is documented in [`src/config.py`](src/config.py). Two are forced (the
paper is unrunnable as literally written):

* **Encoder depth.** Table 3 lists "encoder layers = 124"; 124 message-passing
  layers oversmooth catastrophically, so `encoder_layers` defaults to **2**.
* **Reparameterization.** The literal formula takes `log()` of raw activations
  (produces NaNs), so a standard `μ`/`logvar` parameterization is used instead.

One is a minor resource concession:

* **Decoder width.** Table 3's decoder hidden size of 64 is halved to
  `decoder_hidden = 32`.

Everything else matches Table 3: 16 heads, dropout 0.4, regularization 0.01, **no
KL term**, and **no latent bottleneck** (latent dim = hidden = 64). The opt-in
training tricks (`--small-train` etc.) are all **off** by default.
