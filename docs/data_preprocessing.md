# Data Preprocessing

How raw credit-card transaction rows become the heterogeneous graph the model
trains on. All code lives in [`src/data_prep.py`](../src/data_prep.py) (the
train/validation split is in [`src/train.py`](../src/train.py)).

Dataset: the Kaggle **`kartik2112/fraud-detection`** ("Sparkov" simulator) —
`fraudTrain.csv` / `fraudTest.csv` — the same data used in the paper.

```
raw CSV rows
   │  1. load + validate + subsample        (resolve_data_paths, load_raw)
   ▼
cleaned DataFrame
   │  2. genuine-only train / validation split   (_split_train, in train.py)
   ▼
train_gen (genuine) · val (genuine + all fraud) · test
   │  3. per-type feature engineering        (TypeFeaturizer.fit / .transform)
   ▼
per-type numeric feature matrices
   │  4. graph construction                  (build_hetero_data)
   ▼
HeteroData  (customer / merchant / transaction nodes + typed edges)
   │  5. auxiliary artifacts for the demo    (reference store, demo aux, data profile)
   ▼
everything saved into outputs/artifacts.pt
```

---

## 1. Loading & cleaning — `resolve_data_paths`, `load_raw`

**Locate the files.** `resolve_data_paths` looks for the real Kaggle CSVs in
`data/` or `archive/` and raises a clear `FileNotFoundError` if they are missing.
There is no synthetic fallback — the real dataset is required.

**Load & validate.** `load_raw`:
- reads the CSV and drops the leading unnamed row-index column the Kaggle files ship with;
- checks that every required column is present, raising `ValueError` if not. Required columns:

  ```
  trans_date_trans_time, cc_num, merchant, category, amt, gender,
  city_pop, lat, long, merch_lat, merch_long, dob         (+ is_fraud, trans_num)
  ```

**Subsample from the real data.** If a size limit is given (`--train-size` /
`--test-size`, i.e. `cfg.train_subsample` / `cfg.test_subsample`), it draws that
many rows **while keeping every fraud row** (`keep_all_fraud=True`): it takes all
fraud plus a random genuine pool to fill the quota, then `reset_index`.

> **Consequence for class balance.** Because all fraud is kept, a subsampled
> training set has an *inflated* fraud fraction versus the full file. This is
> intentional (the auto-encoder trains only on genuine rows; fraud is reserved for
> validation), but it means the sampled counts are not the true base rate. The
> distribution *shapes* (amount, hour, category) remain representative.

The two identifier fields `cc_num`, `merchant` (and `trans_num`) are used **only**
to define graph nodes/connectivity — never as model features. `is_fraud` is
excluded from all features and used only for thresholding and evaluation.

## 2. Genuine-only train / validation split — `_split_train`

The model learns "what normal looks like," so it trains on **genuine transactions
only**:

- `train_gen` = genuine rows minus a held-out fraction (`val_fraction`, default 0.15).
- `val` = the held-out genuine rows **+ all fraud** rows from the training file.

Fraud never enters the training forward/backward pass; it is used only to pick the
decision threshold, for early stopping (validation AUC-PR), and for test metrics.
The featurizers (next stage) are **fit on `train_gen` only**, so no statistics leak
from validation, test, or fraud.

## 3. Per-type feature engineering — `TypeFeaturizer`

The graph has three node types, and — following the paper — **each type has its own
attribute set** of different dimensionality. One `TypeFeaturizer` per type handles
it, in three phases (`_raw` → `fit` → `transform`).

### 3a. Derive clean features (`_raw`)

Messy CSV columns become clean numeric/categorical columns, per type:

| Node type | Feature | Kind | Derivation |
|-----------|---------|------|-----------|
| **transaction** | `log_amt` | continuous | `log1p(amt)` — compresses the heavy-tailed amount |
| | `hour_sin`, `hour_cos` | continuous | hour-of-day encoded **cyclically** (so 23:00 ≈ 00:00) |
| | `dow_sin`, `dow_cos` | continuous | day-of-week encoded cyclically |
| | `log_distance` | continuous | `log1p(haversine(lat,long → merch_lat,merch_long))` (`_haversine_km`) |
| **customer** | `age` | continuous | transaction year − `dob` year |
| | `log_city_pop` | continuous | `log1p(city_pop)` |
| | `home_lat`, `home_long` | continuous | cardholder coordinates |
| | `gender` | categorical | M / F |
| **merchant** | `merch_lat_f`, `merch_long_f` | continuous | merchant coordinates |
| | `category` | categorical | merchant category |

All numeric parsing uses `errors="coerce"` + `fillna`, so a malformed field is
coerced/filled rather than crashing inference. The exact column sets are the module
constants `CUST_CONT`/`CUST_CAT`, `MERC_CONT`/`MERC_CAT`, `TXN_CONT`/`TXN_CAT`.

### 3b. Fit (`fit`) — on genuine training rows only

- **Continuous:** compute each column's mean and standard deviation (with a `1e-8`
  floor so a constant column can't divide by zero).
- **Categorical:** record the sorted vocabulary of each categorical column.

### 3c. Transform (`transform` / `transform_one`)

- **Continuous** → standardized `(x − mean) / std`.
- **Categorical** → one-hot against the fitted vocabulary; an **unknown class at
  inference → all-zeros** (no crash, no fake category).
- The output vector is **continuous columns first, then one-hot blocks** — a fixed
  layout the loss and scorer rely on (continuous targets are `x[:, :n_cont]`, each
  categorical block is `x[:, start:start+width]`).

`type_targets_from_featurizers` packages this layout as
`{type: {"cont": n_cont, "cat_groups": [(name, start, width), ...]}}` for the loss
and scoring code.

**Single-record computability & cold-start.** Every feature derives from one raw
record, so `transform_one(record)` produces exactly what `transform` would for that
row. This is what lets the live demo featurize a brand-new customer/merchant from
the incoming transaction alone (cold-start) with the identical transform used in
training.

## 4. Graph construction — `build_hetero_data`

Turns a featurized DataFrame into a PyG `HeteroData` object.

- **Nodes.** `np.unique(cc_num)` → customer nodes, `np.unique(merchant)` → merchant
  nodes, one transaction node per row. Customer/merchant static attributes come
  from de-duplicated rows, transformed by their own featurizer and **aligned to the
  unique-node order** (ids are cast to `str` and reordered with `.loc[unique]`, so
  node *i* corresponds to the right entity and to the edge indices).
- **Node features.** Each type gets its own feature matrix (customer demographics,
  merchant category/location, transaction amount/time/distance) — the differing
  dimensionalities the type-specific encoder projections consume.
- **Labels.** `is_fraud` is attached to transaction nodes only:
  `data["transaction"].y`.
- **Edges (bidirectional).** `customer → transaction` (`makes`) and
  `merchant → transaction` (`sells`), plus their reverses (`rev_makes`,
  `rev_sells`). So a transaction aggregates from its customer and merchant, and
  those nodes receive context in return. (The paper's `Pays`/`Receives`/`Refunds`
  naming is not modeled literally; `Refunds` is absent from the Sparkov schema.)

The canonical node/edge type lists are the `HETERO_METADATA` constant, reused by the
model and the single-transaction scorer so the graph schema is identical at train
and inference time.

## 5. Auxiliary artifacts (saved for the demo)

Computed once during training and stored in `outputs/artifacts.pt`:

- **Reference store** (`compute_reference_store`): each genuine customer's / merchant's
  transformed feature vector, so the demo can look up a *known* entity's profile
  (unknown → cold-start from the record).
- **Demo aux** (`build_demo_aux`): a handful of real example rows + entity
  attributes so the UI can autofill realistic transactions.
- **Data profile** (`compute_data_profile`): a compact EDA summary — class balance,
  a log-amount histogram (genuine vs fraud), fraud rate by hour, and fraud rate by
  top merchant category — rendered by the demo's *Data characteristics* panel
  without re-reading the CSVs.

## Design principles (do not "fix" these — they are deliberate)

1. **Single-record computable.** Every feature derives from one raw record, so the
   training transform and the live demo transform are identical.
2. **Fit on genuine training rows only.** No statistics leak from validation, test,
   or fraud.
3. **Per-type attributes.** Customer, merchant, and transaction each have their own
   feature set and dimensionality — this is what makes the encoder's type-specific
   projections meaningful.
4. **Continuous-first layout.** Continuous features precede one-hot blocks in every
   vector; the loss and scorer depend on this ordering.
5. **Identifiers are structure, not features.** `cc_num` / `merchant` / `trans_num`
   define nodes and edges only; `is_fraud` is for calibration/evaluation only.

## Where to look in the code

| Step | Function / symbol | File |
|------|-------------------|------|
| Locate CSVs | `resolve_data_paths` | `src/data_prep.py` |
| Load, validate, subsample | `load_raw` | `src/data_prep.py` |
| Genuine-only split | `_split_train` | `src/train.py` |
| Feature derivation | `TypeFeaturizer._raw` | `src/data_prep.py` |
| Fit stats / vocab | `TypeFeaturizer.fit` | `src/data_prep.py` |
| Standardize + one-hot | `TypeFeaturizer.transform` / `transform_one` | `src/data_prep.py` |
| Target layout | `type_targets_from_featurizers` | `src/data_prep.py` |
| Build the graph | `build_hetero_data` | `src/data_prep.py` |
| Distance helper | `_haversine_km` | `src/data_prep.py` |
| Demo artifacts | `compute_reference_store`, `build_demo_aux`, `compute_data_profile` | `src/data_prep.py` |
