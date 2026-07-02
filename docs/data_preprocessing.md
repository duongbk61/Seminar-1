# Data Preprocessing

How raw credit-card transaction rows are turned into model-ready features. Code:
[`src/data_prep.py`](../src/data_prep.py). Dataset: Kaggle
`kartik2112/fraud-detection` (`fraudTrain.csv` / `fraudTest.csv`).

## 1. Load & clean (`load_raw`)

- Read the CSV; drop the leading unnamed row-index column.
- Validate that all required columns exist (`amt`, `trans_date_trans_time`,
  `cc_num`, `merchant`, `category`, `gender`, `city_pop`, `lat`, `long`,
  `merch_lat`, `merch_long`, `dob`, `is_fraud`, `trans_num`).
- Optionally subsample to `--train-size` / `--test-size` rows, **keeping every
  fraud row** (`keep_all_fraud`).
- Missing/bad values are coerced (`errors="coerce"`) and filled, so a malformed
  field never crashes preprocessing.

Identifiers (`cc_num`, `merchant`, `trans_num`) are used only to define graph
nodes — never as features. `is_fraud` is used only for evaluation/thresholding.

## 2. Preprocessing methods

Each raw field is converted with the method that fits its meaning
(`TypeFeaturizer._raw`):

| Method | Applied to | Why |
|--------|-----------|-----|
| **Log transform** `log1p(x)` | `amt` → `log_amt`, `city_pop` → `log_city_pop`, distance → `log_distance` | compress heavy right-skewed values |
| **Cyclical encoding** `sin/cos(2π·v/period)` | hour-of-day, day-of-week | makes 23:00 ≈ 00:00 and Sun ≈ Mon (no false "distance" at the wrap-around) |
| **Haversine distance** | `(lat,long)` vs `(merch_lat,merch_long)` | great-circle km between cardholder and merchant, then log-transformed |
| **Age derivation** | `trans_date` year − `dob` year | turns a birth date into a usable number |
| **Standardization (z-score)** `(x − μ)/σ` | all continuous features | zero mean / unit variance so no feature dominates by scale (σ floored at 1e-8) |
| **One-hot encoding** | `gender`, `category` | categorical → numeric; unknown class at inference → all-zeros |

**Fit vs transform.** `fit` computes the standardization μ/σ and the categorical
vocabularies **on genuine training rows only** (no leakage from validation, test,
or fraud). `transform` then applies those fixed statistics to any rows;
`transform_one` does the same for a single record, so the live demo uses the exact
same preprocessing as training.

## 3. Per-type feature vectors

The three node types have **different attribute sets**, so each has its own
featurizer:

| Node type | Continuous (standardized) | Categorical (one-hot) |
|-----------|---------------------------|-----------------------|
| **transaction** | `log_amt`, `log_distance`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | — |
| **customer** | `age`, `log_city_pop`, `home_lat`, `home_long` | `gender` |
| **merchant** | `merch_lat`, `merch_long` | `category` |

**Output layout:** each vector is **continuous columns first, then the one-hot
blocks**. This ordering is a contract the loss and scorer rely on (continuous
targets = the first `n_cont` columns; each categorical block = a known slice).

The resulting per-type matrices become the node features of the heterogeneous
graph (`build_hetero_data`), with `is_fraud` attached to the transaction nodes.
