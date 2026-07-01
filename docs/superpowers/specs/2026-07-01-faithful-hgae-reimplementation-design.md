# Faithful Re-implementation of the HGAE Paper

**Date:** 2026-07-01
**Status:** Approved design → pending spec review
**Goal:** Change `src/` so the code implements *Heterogeneous Graph Auto-Encoder
for Credit Card Fraud Detection* (Majumder et al., IJCA 32(2) 2025) **exactly as
described in the paper and the review report**, reverting the deliberate
deviations currently documented in `CLAUDE.md`.

---

## 1. Motivation & scope

The current code is a *tuned* re-implementation that deliberately diverges from
the paper (selective 3-feature reconstruction, F1-sweep threshold, added KL term,
12-dim latent bottleneck, 8 heads, dropout 0.1, weight-decay 1e-5, shared node
feature space). The user wants the code to match the paper faithfully.

**Stance:** implement the paper literally, deviating **only** where the paper is
unrunnable, and documenting each forced deviation in-code. Every deviation the
paper does *not* force is reverted to the paper's value.

**Forced deviations (the only ones allowed), per user decisions:**

| Paper says | We do | Why |
|---|---|---|
| 124 encoder layers (Table 3) | `encoder_layers = 2` | 124 message-passing layers oversmooth catastrophically; embeddings collapse. |
| Reparam `h = mean(h) + ε·exp(½·log(h))` on raw activations | Standard `mu`/`logvar` linear heads | `log()` of raw (possibly negative) activations → NaN. Standard VAE reparam is the numerically-stable equivalent. |

Everything else follows the paper. Dataset default stays the CPU-friendly
subsample; `--full` remains available (unchanged behaviour).

**In scope:** `src/data_prep.py`, `src/model.py`, new `src/hgae_conv.py`,
`src/train.py`, `src/evaluate.py`, `src/scorer.py`, `src/config.py`,
`make_sample_data.py` (only if new columns are needed), `CLAUDE.md`, tests.
**Out of scope:** the LaTeX report, `app.py` UI layout (interface unchanged),
graph partitioning / community detection / neighbour sampling (paper mentions but
does not specify; remains a noted limitation).

---

## 2. Heterogeneous node attributes (`data_prep.py`)

**Current:** all three node types share one ~23-dim transaction feature vector;
customer/merchant features = mean of attached transactions.

**Target (paper §4.1, report §Data Representation):** each node type has its own
attribute set of **differing dimensionality**, reconciled by the encoder's
type-specific projections.

Per-type featurizers (each fit on genuine training rows only, each single-record
computable so inference reuses the identical transform):

- **Customer** (keyed by `cc_num`): `gender_M`, `age`, `log_city_pop`,
  home `lat`/`long` (standardized continuous + one-hot gender).
- **Merchant** (keyed by `merchant`): `category` (one-hot), `merch_lat`,
  `merch_long`.
- **Transaction** (keyed by `trans_num`): `log_amt`, `hour_sin`, `hour_cos`,
  `dow_sin`, `dow_cos`, `log_distance`.

Each type records which columns are **continuous** (MSE target) vs. **categorical**
(cross-entropy target) so the decoder/loss can treat them correctly.

Node identity fields (`cc_num`, `merchant`, `trans_num`) are used only for graph
connectivity, never as features. `is_fraud` is excluded from all features and used
only for threshold calibration + evaluation.

**Graph edges:** transaction `dt` aggregates from its source nodes
`D(dt) = {customer, merchant}` (paper Eq. 1). We keep **bidirectional** edges so
customer and merchant nodes also receive contextualized embeddings (needed because
the loss sums over all node types):
`customer → transaction`, `merchant → transaction`, plus reverse edges.
The paper's `Pays`/`Receives`/`Refunds` relation naming is noted; `Refunds` is not
modeled (absent from the Sparkov schema). Document this.

**Node features:** each type's node feature matrix comes from its **own**
featurizer (NOT the mean-of-transactions shortcut). A customer node's features are
that customer's demographics; a merchant node's are that merchant's category/loc.
For customers/merchants that appear on multiple rows, use the (identical) static
attributes from any of their rows.

---

## 3. Custom intra-edge attention layer (`src/hgae_conv.py`)

New module replacing PyG `HGTConv`. Implements paper Eq. 2–5 exactly.

For a destination node `dt` of type `t_d`, over each relation `r` with source type
`t_s`, using `H` heads (dim per head `dim_k = hidden // H`):

- Per-head attention scalar (Eq. 4):
  `Att^k(v, e_r, d) = Linear^{S_k}_{t_s}(h_v) · W^{Att}_r · (Linear^{D_k}_{t_d}(h_d))^T`
  → one scalar per (edge, head).
- **Softmax across the H heads of that single edge** (Eq. 3):
  `f_Attent = softmax_k( [Att^1, …, Att^H] )` → H weights per edge summing to 1.
  (NOT softmax across neighbours — this is the paper's distinctive choice.)
- Per-head message (Eq. 5):
  `f_Mssg^k(v, e_r, d) = Linear^{M_k}_{t_s}(h_v) · W^{Mssg}_r` → concat over k → dim `hidden`.
- Edge contribution: scale head-block k of `f_Mssg` by attention weight k, i.e.
  `concat_k( f_Attent^k · f_Mssg^k )`.
- **Uniform aggregation over neighbours** (Eq. 2, ⊕ = sum, no neighbour
  normalization): `f_enc(dt) = Σ_{v ∈ D(dt)} edge_contribution(v)` summed across
  all relations feeding `dt`.
- Output (Eq. 1): `Linear_{t_d}( f_enc(dt) ) ⊕ h⁰_{dt}` (residual add of the
  layer-0 projected features of `dt`).

Learnable per relation `r`: `W_Att_r ∈ R^{dim_k × dim_k}`, `W_Mssg_r ∈ R^{dim_k × dim_k}`.
Learnable per (type, head): source/dest/message linear projections.

Implementation: a `torch.nn.Module` operating on a PyG `HeteroData`
`x_dict`/`edge_index_dict`; per-relation message computation via `scatter_add` on
destination indices (uniform sum). Must run on CPU.

heads = **16** (Table 3).

---

## 4. Encoder / reparameterization / decoder / loss (`model.py`)

**Encoder:** type-specific input projection `h⁰ = Linear_t(x_t)` → stack of
`encoder_layers` (=2) `HGAEConv` layers. After each layer, apply reparameterization
to every node type (paper Algorithm 1 line 7).

**Reparameterization:** per node type, `mu = Linear(h)`, `logvar = Linear(h)`,
`z = mu + ε·exp(½·logvar)` at train, `z = mu` at eval (stable). latent dim =
hidden = **64** (paper has **no bottleneck**).

**Decoder (Eq. 6):** per-type MLP `f_dec_t` reconstructing that type's **full
attribute vector**. Continuous head → linear (MSE); categorical head(s) → logits
per categorical group (cross-entropy).

**Loss (Eq. 7–8):** for each node type t:
`L_t = MSE(continuous) + Σ_groups CrossEntropy(categorical_group)`, averaged over
that type's nodes; total `L = Σ_t w_t · L_t` where `w_t` = prevalence weight
(fraction of nodes of type t). **No KL term** (β removed).

**Anomaly score:** per transaction, its combined reconstruction loss
(continuous MSE summed with categorical CE, over the transaction attribute set),
computed deterministically at eval. This is the value thresholded.

`ae_loss` signature changes: it now consumes per-type reconstructions, targets, and
categorical-group metadata; returns total loss + per-type breakdown for logging.
No `mu`/`logvar` in the loss (KL gone) — they exist only for the sampling noise.

---

## 5. Threshold — both rules (`evaluate.py`, `train.py`)

- **Decision rule (used for the verdict):** `Threshold = μ + 2σ` where μ, σ are the
  mean and std of reconstruction errors on a **held-out set of genuine
  transactions** (paper Eq. 9, report pipeline figure). New function
  `mu_2sigma_threshold(genuine_scores)`.
- **Also computed and logged for comparison:** the existing F1-max sweep
  (`search_threshold`) operating point on the validation set (paper Fig. 5c).
- `train.py` saves both thresholds in artifacts + `metrics.json`; the μ+2σ value is
  the one `scorer` uses. Test metrics reported at the μ+2σ operating point, with the
  F1-sweep metrics logged alongside.

---

## 6. Config (`config.py`)

Faithful values (Table 3) + the one forced-deviation docstring:

```
hidden_dim      = 64      # Table 3 "Size of Hidden Layers"
heads           = 16      # Table 3 "Number of heads (H)"
encoder_layers  = 2       # FORCED DEVIATION: Table 3 says 124 -> oversmooths; 2 is runnable
decoder_hidden  = 64      # Table 3 "Number of Layers for the Decoder" read as width
latent_dim      = 64      # = hidden; paper has NO bottleneck
dropout         = 0.4     # Table 3 "Dropout Rate"
weight_decay    = 0.01    # Table 3 "Regularization Rate"
beta            = 0.0     # paper has NO KL term
```

Fix the docstring's table reference: hyperparameters are the paper's **Table 3**
(Table 4 is the results table). Subsample default unchanged; `--full` unchanged.

---

## 7. Scorer + demo + docs

- **`scorer.py`:** build the 3-node graph (1 transaction + its customer + its
  merchant) using the **type-specific** featurizers. Score = the transaction's
  combined reconstruction loss vs. the μ+2σ threshold. Cold-start: derive each
  type's vector from the incoming record's own fields (customer demographics,
  merchant category/loc, transaction amt/time). Reference store holds per-entity
  static attributes per type.
- **`app.py`:** interface unchanged. "Top contributing features" may now include
  categorical attributes; ensure the display handles both continuous and
  categorical reconstruction contributions.
- **`CLAUDE.md`:** rewrite the "Project-specific conventions" and "Deviations"
  sections to describe the faithful implementation and the single remaining forced
  deviation (encoder depth). Remove now-false claims (selective reconstruction, F1
  threshold as the rule, added KL, latent bottleneck).

---

## 8. Testing

New/updated tests under `tests/` (create if absent):

1. **Featurizer round-trip:** each per-type featurizer is single-record computable;
   `transform_one` == `transform` on a 1-row frame; fit uses genuine rows only.
2. **`HGAEConv` correctness:** on a tiny hand-built 2-edge graph, verify (a)
   per-edge attention weights sum to 1 across heads, (b) aggregation is a plain sum
   over neighbours (feed 2 identical neighbours → output = 2× single-neighbour
   contribution before residual), (c) output shape per type.
3. **Loss:** categorical groups use cross-entropy, continuous use MSE; prevalence
   weights sum to 1; no KL contribution.
4. **Threshold:** `mu_2sigma_threshold` == mean+2·std of the genuine scores;
   ~95% of genuine below it on a Gaussian sample.
5. **End-to-end smoke:** `run_training` on synthetic sample data completes, saves
   artifacts, and `FraudScorer.load(...).score_transaction(...)` returns a verdict.

`make_sample_data.py` must still produce all columns the per-type featurizers need
(it already emits the full Sparkov schema — verify, extend only if a needed column
is missing).

---

## 9. Risks / expected outcome

- Substantial rewrite of `data_prep.py`, `model.py`, `scorer.py` + new custom
  layer; each needs tests before wiring together.
- The paper's uniform-neighbour-sum attention has no neighbour normalization;
  combined with no KL and no bottleneck, **detection quality may drop** vs. the
  current tuned version. This is the expected price of fidelity and will be
  documented, not "fixed".
- Numerical stability of the custom attention on CPU (softmax over few heads is
  cheap; watch for exploding sums with uniform aggregation on high-degree nodes —
  mitigated because customer/merchant fan-in is bounded per single-transaction
  inference and manageable on the subsample).
