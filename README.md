# Heterogeneous Graph Auto-Encoder for Credit Card Fraud Detection

Implementation + interactive demo of:

> M. T. Singh et al., *Heterogeneous Graph Auto-Encoder for Credit Card Fraud
> Detection*, arXiv:2410.08121 (2024).

The idea: model transactions as a **heterogeneous graph** (customer ↔ transaction
↔ merchant), encode it with an **attention GNN** (Heterogeneous Graph Transformer),
add a **variational** latent layer, and **reconstruct** transactions with an MLP
decoder. The auto-encoder is trained only on **genuine** transactions, so a high
reconstruction error flags an anomaly ⇒ fraud. This sidesteps the class-imbalance
problem without over/under-sampling.

## Why this design is demo-friendly

Training and inference are decoupled:

* **Train once** → save `outputs/artifacts.pt` (weights, featuriser, decision
  threshold, customer/merchant profiles).
* **Demo** loads those artifacts and scores any *single* new transaction
  instantly on CPU by building a tiny 3-node graph (transaction + its customer +
  its merchant). New customers/merchants are handled via cold-start. **No
  retraining needed to demo.**

## Project layout

```
SeminarProject/
├─ data/                 # put fraudTrain.csv / fraudTest.csv here (see data/README.md)
├─ src/
│  ├─ config.py          # all hyper-parameters (Table 4 of the paper)
│  ├─ data_prep.py       # feature engineering + hetero-graph construction
│  ├─ model.py           # HGT encoder + VAE latent + MLP decoder
│  ├─ train.py           # train-once pipeline, saves artifacts + plots
│  ├─ evaluate.py        # threshold search, metrics, Figure-5 plots
│  ├─ scorer.py          # FraudScorer: real-time single-transaction scoring
│  └─ utils.py
├─ make_sample_data.py   # synthetic Sparkov-schema data (to run without Kaggle)
├─ main.py               # `python main.py` → train
├─ app.py                # `streamlit run app.py` → demo
└─ requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Quick start (no dataset needed)

```bash
python make_sample_data.py      # writes synthetic data/sample/*.csv
python main.py --epochs 40      # trains, evaluates, saves outputs/artifacts.pt
streamlit run app.py            # open the demo in the browser
```

## With the real dataset

1. Download from Kaggle (kartik2112/fraud-detection) and place `fraudTrain.csv`
   and `fraudTest.csv` in `data/` (see `data/README.md`).
2. `python main.py` (subsamples ~120k/40k by default; `--full` for everything).
3. `streamlit run app.py`.

The real files are detected automatically and take priority over the synthetic
sample data.

## Outputs

* `outputs/artifacts.pt` – everything the demo needs.
* `outputs/metrics.json` – test F1 / precision / recall / AUC-PR / ROC-AUC.
* `outputs/plots/*.png` – training loss, score distribution, F1-vs-threshold,
  ROC and Precision-Recall curves (paper Figure 5).

## Notes / deviations from the paper

* Table 4 lists "encoder layers = 124" and "decoder = 64"; 124 message-passing
  layers would over-smooth a GNN, so we use a sane `encoder_layers` (default 2)
  and read "64" as the decoder hidden width. Both are configurable in
  `src/config.py`.
* `heads` defaults to 8 (paper says 16) for speed on CPU; change it freely.
* GPU (e.g. GTX 1650) is used automatically when available; otherwise CPU.
