# Faithful HGAE Re-implementation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `src/` so the code implements the HGAE paper (Majumder et al., IJCA 32(2) 2025) exactly as described in the paper/report, reverting the tuned deviations.

**Architecture:** Heterogeneous graph with per-type node attributes → a custom intra-edge attention encoder (softmax across heads, uniform neighbour sum) → per-type VAE-style reparameterization (no KL) → per-type MLP decoders reconstructing full attribute vectors (MSE for continuous, cross-entropy for categorical). Anomaly = a transaction's combined reconstruction error; verdict = error ≥ μ+2σ of genuine errors.

**Tech Stack:** Python, PyTorch, PyTorch Geometric (`HeteroData`, `MessagePassing`/manual scatter), NumPy, pandas, scikit-learn, pytest.

## Global Constraints

- **Faithful hyperparameters (paper Table 3):** `hidden_dim=64`, `heads=16`, `decoder_hidden=64`, `dropout=0.4`, `weight_decay=0.01`.
- **No KL term:** `beta=0.0`; the loss is reconstruction-only.
- **No latent bottleneck:** `latent_dim=64` (= hidden).
- **Only two forced deviations, each documented in-code:** `encoder_layers=2` (Table 3's 124 oversmooths) and standard `mu`/`logvar` reparameterization (the literal `log(h)` formula NaNs).
- **Threshold:** decision rule is `μ+2σ` on held-out genuine transaction errors; the F1-sweep operating point is also computed and logged for comparison.
- **Continuous features are always the FIRST columns of every node feature vector**, followed by categorical one-hot blocks (relied on by the loss/scoring).
- **Featurizers fit on genuine training rows only**; every feature is single-record computable so inference reuses the identical transform.
- **Run on CPU.** Subsample default unchanged; `--full` unchanged.
- Tests run from the project root with `python -m pytest`. `src/` is an importable package (`src/__init__.py` exists).

---

### Task 1: Environment, config, and test scaffold

**Files:**
- Modify: `requirements.txt`
- Modify: `src/config.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `Config` dataclass with faithful fields `hidden_dim, heads, encoder_layers, decoder_hidden, latent_dim, dropout, weight_decay, beta` and `get_config()`.

- [ ] **Step 1: Add pytest to requirements**

Append a line to `requirements.txt`:

```
pytest
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: torch, torch_geometric, pytest install successfully. Verify:
`python -c "import torch, torch_geometric, pytest; print('ok')"` → prints `ok`.

- [ ] **Step 3: Write the failing config test**

Create `tests/__init__.py` (empty file), then `tests/test_config.py`:

```python
from src.config import get_config


def test_faithful_hyperparameters():
    c = get_config()
    assert c.hidden_dim == 64
    assert c.heads == 16
    assert c.decoder_hidden == 64
    assert c.dropout == 0.4
    assert c.weight_decay == 0.01
    assert c.beta == 0.0          # no KL term
    assert c.latent_dim == 64     # no bottleneck
    assert c.encoder_layers == 2  # forced deviation from Table 3's 124
    assert c.hidden_dim % c.heads == 0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL (current values are heads=8, dropout=0.1, beta=5e-4, latent_dim=12, weight_decay=1e-5).

- [ ] **Step 5: Update `src/config.py`**

Replace the module docstring and the model/optimisation fields. New docstring:

```python
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
```

Set the fields (keep the data-file, subsample, seed, device, output_dir fields as they are):

```python
    # ----- model (paper Table 3) -----
    hidden_dim: int = 64          # "Size of Hidden Layers" = 64
    heads: int = 16               # "Number of heads (H)" = 16
    encoder_layers: int = 2       # FORCED DEVIATION: Table 3 says 124 -> oversmooths
    decoder_hidden: int = 64      # "Number of Layers for the Decoder" = 64 (width)
    latent_dim: int = 64          # = hidden_dim; the paper has NO bottleneck
    dropout: float = 0.4          # "Dropout Rate" = 0.4

    # ----- optimisation -----
    lr: float = 2e-3
    weight_decay: float = 0.01    # "Regularization Rate" = 0.01
    epochs: int = 150
    beta: float = 0.0             # the paper has NO KL term (reconstruction-only)
    val_fraction: float = 0.15
    early_stop_patience: int = 25
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: faithful config (Table 3) + pytest scaffold"
```

---

### Task 2: Per-type featurizers (`data_prep.py`)

**Files:**
- Modify: `src/data_prep.py` (replace `Featurizer`, `CONT_COLS`/`CYC_COLS`/`RECON_COLS`; keep `resolve_data_paths`, `load_raw`, `_haversine_km`, `_mean_by_group`, `build_demo_aux`, `_py`)
- Test: `tests/test_featurizers.py`

**Interfaces:**
- Produces:
  - `TypeFeaturizer(kind, cont_cols, cat_cols, cat_vocab={}, mean_=None, scale_=None)` with methods `fit(df)`, `transform(df)->np.ndarray[float32]`, `transform_one(record:dict)->np.ndarray`, and properties `dim:int`, `cont_idx:list[int]`, `cat_groups:list[tuple[str,int,int]]` (name, start, width), `feature_names:list[str]`.
  - `make_featurizers() -> dict[str, TypeFeaturizer]` with keys `"customer"`, `"merchant"`, `"transaction"`.
  - `type_targets_from_featurizers(feats) -> dict[str, dict]` where each value is `{"cont": int, "cat_groups": list[tuple[str,int,int]]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_featurizers.py`:

```python
import numpy as np
import pandas as pd

from src.data_prep import make_featurizers, type_targets_from_featurizers


def _rows(n=8):
    return pd.DataFrame({
        "trans_date_trans_time": ["2020-06-15 14:00:00"] * n,
        "cc_num": [f"c{i%3}" for i in range(n)],
        "merchant": [f"m{i%2}" for i in range(n)],
        "category": ["grocery_pos", "shopping_net"] * (n // 2),
        "amt": np.linspace(5, 500, n),
        "gender": ["M", "F"] * (n // 2),
        "city_pop": np.linspace(1000, 900000, n),
        "lat": np.linspace(30, 45, n), "long": np.linspace(-120, -75, n),
        "merch_lat": np.linspace(31, 46, n), "merch_long": np.linspace(-119, -74, n),
        "dob": ["1980-01-01"] * n, "is_fraud": [0] * n,
        "trans_num": [f"t{i}" for i in range(n)],
    })


def test_transaction_featurizer_shapes_and_cont_first():
    feats = make_featurizers()
    df = _rows()
    feats["transaction"].fit(df)
    X = feats["transaction"].transform(df)
    f = feats["transaction"]
    assert X.shape == (len(df), f.dim)
    assert X.dtype == np.float32
    # transaction has 6 continuous, 0 categorical
    assert f.cont_idx == [0, 1, 2, 3, 4, 5]
    assert f.cat_groups == []


def test_categorical_group_layout():
    feats = make_featurizers()
    df = _rows()
    feats["merchant"].fit(df)
    f = feats["merchant"]
    # merchant: 2 continuous (merch_lat, merch_long) then one 'category' one-hot block
    assert len(f.cont_cols) == 2
    (name, start, width), = f.cat_groups
    assert name == "category"
    assert start == 2
    assert width == len(f.cat_vocab["category"])


def test_transform_one_matches_transform():
    feats = make_featurizers()
    df = _rows()
    for k in feats:
        feats[k].fit(df)
    rec = df.iloc[0].to_dict()
    for k in ("customer", "merchant", "transaction"):
        a = feats[k].transform(df.iloc[[0]])[0]
        b = feats[k].transform_one(rec)
        assert np.allclose(a, b), k


def test_type_targets_structure():
    feats = make_featurizers()
    df = _rows()
    for k in feats:
        feats[k].fit(df)
    tt = type_targets_from_featurizers(feats)
    assert tt["transaction"]["cont"] == 6
    assert tt["transaction"]["cat_groups"] == []
    assert tt["customer"]["cat_groups"][0][0] == "gender"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_featurizers.py -v`
Expected: FAIL with ImportError (`make_featurizers` not defined).

- [ ] **Step 3: Implement in `src/data_prep.py`**

Update the module docstring's "three node types" note to say each type now has its own attribute set. Remove `CONT_COLS`, `CYC_COLS`, `RECON_COLS` and the old `Featurizer` class + old `build_hetero_data`/`compute_reference_store` (they are replaced here and in Task 3). Keep `REQUIRED_COLS`, `resolve_data_paths`, `load_raw`, `_haversine_km`, `_mean_by_group`, `build_demo_aux`, `_py`. Add:

```python
# ----- per-type attribute definitions (paper: heterogeneous attribute sets) -----
CUST_CONT = ["age", "log_city_pop", "home_lat", "home_long"]
CUST_CAT = ["gender"]
MERC_CONT = ["merch_lat_f", "merch_long_f"]
MERC_CAT = ["category"]
TXN_CONT = ["log_amt", "log_distance", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TXN_CAT: list[str] = []


@dataclass
class TypeFeaturizer:
    """Fit on genuine rows; transform any records (incl. one) for ONE node type.

    Layout of the output vector: standardized continuous columns FIRST, then one
    one-hot block per categorical column (order = `cat_cols`).
    """
    kind: str
    cont_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    cat_vocab: dict = field(default_factory=dict)      # name -> sorted class list
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def _raw(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        if self.kind == "customer":
            ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
            dob = pd.to_datetime(df["dob"], errors="coerce")
            age = (ts.dt.year - dob.dt.year)
            out["age"] = age.fillna(age.median() if age.notna().any() else 40).astype(float)
            pop = pd.to_numeric(df["city_pop"], errors="coerce").fillna(0.0)
            out["log_city_pop"] = np.log1p(pop.clip(lower=0))
            out["home_lat"] = pd.to_numeric(df["lat"], errors="coerce").fillna(0.0)
            out["home_long"] = pd.to_numeric(df["long"], errors="coerce").fillna(0.0)
            out["gender"] = df["gender"].astype(str).str.upper().values
        elif self.kind == "merchant":
            out["merch_lat_f"] = pd.to_numeric(df["merch_lat"], errors="coerce").fillna(0.0)
            out["merch_long_f"] = pd.to_numeric(df["merch_long"], errors="coerce").fillna(0.0)
            out["category"] = df["category"].astype(str).values
        else:  # transaction
            amt = pd.to_numeric(df["amt"], errors="coerce").fillna(0.0)
            out["log_amt"] = np.log1p(amt.clip(lower=0))
            ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
            hour = ts.dt.hour.fillna(12).astype(float)
            dow = ts.dt.dayofweek.fillna(0).astype(float)
            out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
            out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
            dist = _haversine_km(
                pd.to_numeric(df["lat"], errors="coerce").fillna(0.0).values,
                pd.to_numeric(df["long"], errors="coerce").fillna(0.0).values,
                pd.to_numeric(df["merch_lat"], errors="coerce").fillna(0.0).values,
                pd.to_numeric(df["merch_long"], errors="coerce").fillna(0.0).values,
            )
            out["log_distance"] = np.log1p(np.clip(dist, 0, None))
        return out

    def fit(self, df: pd.DataFrame) -> "TypeFeaturizer":
        raw = self._raw(df)
        cont = raw[self.cont_cols].to_numpy(dtype=np.float64)
        self.mean_ = cont.mean(axis=0)
        std = cont.std(axis=0)
        std[std < 1e-8] = 1.0
        self.scale_ = std
        for c in self.cat_cols:
            self.cat_vocab[c] = sorted(raw[c].astype(str).unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._raw(df)
        cont = (raw[self.cont_cols].to_numpy(dtype=np.float64) - self.mean_) / self.scale_
        blocks = [cont]
        for c in self.cat_cols:
            vocab = self.cat_vocab[c]
            idx = {v: i for i, v in enumerate(vocab)}
            oh = np.zeros((len(raw), len(vocab)), dtype=np.float64)
            for r, v in enumerate(raw[c].astype(str).values):
                if v in idx:
                    oh[r, idx[v]] = 1.0
            blocks.append(oh)
        return np.hstack(blocks).astype(np.float32)

    def transform_one(self, record: dict) -> np.ndarray:
        return self.transform(pd.DataFrame([record]))[0]

    @property
    def dim(self) -> int:
        return len(self.cont_cols) + sum(len(self.cat_vocab[c]) for c in self.cat_cols)

    @property
    def cont_idx(self) -> list[int]:
        return list(range(len(self.cont_cols)))

    @property
    def cat_groups(self) -> list[tuple[str, int, int]]:
        groups, start = [], len(self.cont_cols)
        for c in self.cat_cols:
            w = len(self.cat_vocab[c])
            groups.append((c, start, w))
            start += w
        return groups

    @property
    def feature_names(self) -> list[str]:
        names = list(self.cont_cols)
        for c in self.cat_cols:
            names += [f"{c}={v}" for v in self.cat_vocab[c]]
        return names


def make_featurizers() -> dict:
    return {
        "customer": TypeFeaturizer("customer", list(CUST_CONT), list(CUST_CAT)),
        "merchant": TypeFeaturizer("merchant", list(MERC_CONT), list(MERC_CAT)),
        "transaction": TypeFeaturizer("transaction", list(TXN_CONT), list(TXN_CAT)),
    }


def type_targets_from_featurizers(feats: dict) -> dict:
    return {t: {"cont": len(f.cont_cols), "cat_groups": f.cat_groups}
            for t, f in feats.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_featurizers.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_prep.py tests/test_featurizers.py
git commit -m "feat: per-type featurizers with continuous+categorical layout"
```

---

### Task 3: Heterogeneous graph + reference store (`data_prep.py`)

**Files:**
- Modify: `src/data_prep.py` (add `build_hetero_data`, `compute_reference_store`, `HETERO_METADATA`)
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `make_featurizers`, `TypeFeaturizer` (Task 2).
- Produces:
  - `build_hetero_data(df, feats) -> (HeteroData, info)`. Node types `customer/merchant/transaction`; each `.x` from its own featurizer; `data["transaction"].y` = labels. Edges: `("customer","makes","transaction")`, `("transaction","rev_makes","customer")`, `("merchant","sells","transaction")`, `("transaction","rev_sells","merchant")`. `info` has keys `labels, n_tx, customer_ids, merchant_ids`.
  - `compute_reference_store(df_genuine, feats) -> dict` with `{"customer": {id: vec}, "merchant": {id: vec}, "customer_dim": int, "merchant_dim": int}`.
  - `HETERO_METADATA = (node_types_list, edge_types_list)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_graph.py`:

```python
import numpy as np
import pandas as pd

from src.data_prep import make_featurizers, build_hetero_data, compute_reference_store


def _rows(n=8):
    return pd.DataFrame({
        "trans_date_trans_time": ["2020-06-15 14:00:00"] * n,
        "cc_num": [f"c{i%3}" for i in range(n)],
        "merchant": [f"m{i%2}" for i in range(n)],
        "category": ["grocery_pos", "shopping_net"] * (n // 2),
        "amt": np.linspace(5, 500, n),
        "gender": ["M", "F"] * (n // 2),
        "city_pop": np.linspace(1000, 900000, n),
        "lat": np.linspace(30, 45, n), "long": np.linspace(-120, -75, n),
        "merch_lat": np.linspace(31, 46, n), "merch_long": np.linspace(-119, -74, n),
        "dob": ["1980-01-01"] * n, "is_fraud": ([0] * (n - 1)) + [1],
        "trans_num": [f"t{i}" for i in range(n)],
    })


def test_build_hetero_data_shapes_and_edges():
    feats = make_featurizers()
    df = _rows()
    for k in feats:
        feats[k].fit(df)
    data, info = build_hetero_data(df, feats)
    assert data["transaction"].x.shape == (8, feats["transaction"].dim)
    assert data["customer"].x.shape == (3, feats["customer"].dim)
    assert data["merchant"].x.shape == (2, feats["merchant"].dim)
    assert data["transaction"].y.shape == (8,)
    # each transaction has exactly one customer and one merchant edge into it
    assert data["customer", "makes", "transaction"].edge_index.shape == (2, 8)
    assert data["merchant", "sells", "transaction"].edge_index.shape == (2, 8)
    assert info["n_tx"] == 8


def test_reference_store_keys():
    feats = make_featurizers()
    df = _rows()
    for k in feats:
        feats[k].fit(df)
    ref = compute_reference_store(df[df.is_fraud == 0], feats)
    assert set(ref["customer"].keys()) <= {"c0", "c1", "c2"}
    vec = next(iter(ref["customer"].values()))
    assert len(vec) == feats["customer"].dim
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Implement in `src/data_prep.py`**

Add:

```python
def build_hetero_data(df: pd.DataFrame, feats: dict):
    """HeteroData with per-type node features; transaction nodes carry labels."""
    import torch
    from torch_geometric.data import HeteroData

    cust_uni, cust_idx = np.unique(df["cc_num"].astype(str).values, return_inverse=True)
    merch_uni, merch_idx = np.unique(df["merchant"].astype(str).values, return_inverse=True)
    tx_idx = np.arange(len(df))

    df_s = df.copy()
    df_s["cc_num"] = df_s["cc_num"].astype(str)
    df_s["merchant"] = df_s["merchant"].astype(str)
    cust_df = df_s.drop_duplicates("cc_num").set_index("cc_num").loc[cust_uni].reset_index()
    merch_df = df_s.drop_duplicates("merchant").set_index("merchant").loc[merch_uni].reset_index()

    X_cust = feats["customer"].transform(cust_df)
    X_merch = feats["merchant"].transform(merch_df)
    X_txn = feats["transaction"].transform(df)

    data = HeteroData()
    data["customer"].x = torch.from_numpy(X_cust)
    data["merchant"].x = torch.from_numpy(X_merch)
    data["transaction"].x = torch.from_numpy(X_txn)
    data["transaction"].y = torch.tensor(df["is_fraud"].to_numpy(dtype=np.int64))

    c2t = np.vstack([cust_idx, tx_idx]).astype(np.int64)
    m2t = np.vstack([merch_idx, tx_idx]).astype(np.int64)
    data["customer", "makes", "transaction"].edge_index = torch.from_numpy(c2t)
    data["transaction", "rev_makes", "customer"].edge_index = torch.from_numpy(c2t[[1, 0]])
    data["merchant", "sells", "transaction"].edge_index = torch.from_numpy(m2t)
    data["transaction", "rev_sells", "merchant"].edge_index = torch.from_numpy(m2t[[1, 0]])

    info = {
        "labels": df["is_fraud"].to_numpy(dtype=np.int64),
        "n_tx": len(df),
        "customer_ids": cust_uni,
        "merchant_ids": merch_uni,
    }
    return data, info


def compute_reference_store(df_genuine: pd.DataFrame, feats: dict) -> dict:
    """Per-entity genuine feature vectors for known-entity lookup in the demo."""
    out = {}
    cust = df_genuine.drop_duplicates("cc_num")
    Xc = feats["customer"].transform(cust)
    out["customer"] = {str(k): Xc[i] for i, k in enumerate(cust["cc_num"].values)}
    merc = df_genuine.drop_duplicates("merchant")
    Xm = feats["merchant"].transform(merc)
    out["merchant"] = {str(k): Xm[i] for i, k in enumerate(merc["merchant"].values)}
    out["customer_dim"] = int(feats["customer"].dim)
    out["merchant_dim"] = int(feats["merchant"].dim)
    return out


HETERO_METADATA = (
    ["customer", "merchant", "transaction"],
    [
        ("customer", "makes", "transaction"),
        ("transaction", "rev_makes", "customer"),
        ("merchant", "sells", "transaction"),
        ("transaction", "rev_sells", "merchant"),
    ],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_prep.py tests/test_graph.py
git commit -m "feat: per-type hetero graph construction + reference store"
```

---

### Task 4: Custom intra-edge attention layer (`src/hgae_conv.py`)

**Files:**
- Create: `src/hgae_conv.py`
- Test: `tests/test_hgae_conv.py`

**Interfaces:**
- Produces:
  - `intra_edge_softmax(att: Tensor[E,H]) -> Tensor[E,H]` — softmax across heads (dim=1).
  - `HGAEConv(metadata, hidden_dim, heads)` `nn.Module`; `forward(x_dict, edge_index_dict) -> dict[str, Tensor]`. Implements paper Eq. 1–5: per-head attention, softmax across heads, uniform (summed) neighbour aggregation, residual add of the layer input. `out_lin` has **no bias** (so aggregation scaling is testable).

- [ ] **Step 1: Write the failing test**

Create `tests/test_hgae_conv.py`:

```python
import torch

from src.data_prep import HETERO_METADATA
from src.hgae_conv import HGAEConv, intra_edge_softmax


def test_intra_edge_softmax_sums_to_one_across_heads():
    att = torch.randn(5, 4)  # 5 edges, 4 heads
    alpha = intra_edge_softmax(att)
    assert alpha.shape == (5, 4)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(5), atol=1e-5)


def _two_node_graph(n_customers):
    # n_customers identical customers all pointing at ONE transaction
    x_dict = {
        "customer": torch.ones(n_customers, 8),
        "merchant": torch.ones(1, 8),
        "transaction": torch.zeros(1, 8),
    }
    c2t = torch.tensor([[i for i in range(n_customers)], [0] * n_customers])
    m2t = torch.tensor([[0], [0]])
    edge_index_dict = {
        ("customer", "makes", "transaction"): c2t,
        ("transaction", "rev_makes", "customer"): c2t[[1, 0]],
        ("merchant", "sells", "transaction"): m2t,
        ("transaction", "rev_sells", "merchant"): m2t[[1, 0]],
    }
    return x_dict, edge_index_dict


def test_uniform_neighbour_aggregation_is_a_sum():
    torch.manual_seed(0)
    conv = HGAEConv(HETERO_METADATA, hidden_dim=8, heads=2)
    conv.eval()
    x1, e1 = _two_node_graph(1)
    x2, e2 = _two_node_graph(2)  # duplicate the identical customer neighbour
    with torch.no_grad():
        out1 = conv(x1, e1)["transaction"]
        out2 = conv(x2, e2)["transaction"]
    # residual is x_dict["transaction"] (zeros) in both; out_lin has no bias, so
    # doubling identical neighbours doubles the transaction output.
    assert torch.allclose(out2, 2 * out1, atol=1e-5)


def test_forward_output_shapes():
    conv = HGAEConv(HETERO_METADATA, hidden_dim=8, heads=2)
    x, e = _two_node_graph(3)
    out = conv(x, e)
    assert out["transaction"].shape == (1, 8)
    assert out["customer"].shape == (3, 8)
    assert out["merchant"].shape == (1, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hgae_conv.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/hgae_conv.py`**

```python
"""Custom intra-edge attention layer for the HGAE encoder (paper Eq. 1-5).

Distinctive to this paper (vs. standard HGT / GAT): the softmax is normalized
ACROSS THE H ATTENTION HEADS of a single edge (`intra_edge_softmax`), and
neighbour messages are aggregated by a UNIFORM SUM (element-wise addition, the
paper's oplus), NOT a neighbour-normalized weighting. Each relation r owns its
learnable W_Att_r and W_Mssg_r; each (node type, head) owns source/dest/message
projections.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _rk(rel) -> str:
    return "__".join(rel)


def intra_edge_softmax(att: torch.Tensor) -> torch.Tensor:
    """Softmax across the H heads of each edge. att: [E, H] -> [E, H]."""
    return torch.softmax(att, dim=1)


class HGAEConv(nn.Module):
    def __init__(self, metadata, hidden_dim: int = 64, heads: int = 16) -> None:
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.node_types, self.edge_types = metadata[0], metadata[1]
        self.hidden = hidden_dim
        self.heads = heads
        self.dk = hidden_dim // heads

        # per-(type) projections producing all heads at once (view to [N,H,dk])
        self.s_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.node_types})
        self.d_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.node_types})
        self.m_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.node_types})
        # Eq. 1 output projection per destination type (no bias -> testable scaling)
        self.out_lin = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim, bias=False) for t in self.node_types})

        # per-relation learnable matrices W_Att_r, W_Mssg_r : [H, dk, dk]
        self.w_att = nn.ParameterDict()
        self.w_mssg = nn.ParameterDict()
        for rel in self.edge_types:
            wa = torch.empty(heads, self.dk, self.dk)
            wm = torch.empty(heads, self.dk, self.dk)
            nn.init.xavier_uniform_(wa)
            nn.init.xavier_uniform_(wm)
            self.w_att[_rk(rel)] = nn.Parameter(wa)
            self.w_mssg[_rk(rel)] = nn.Parameter(wm)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        H, dk = self.heads, self.dk
        # aggregation accumulators per destination type
        agg = {t: x_dict[t].new_zeros(x_dict[t].size(0), H, dk) for t in x_dict}

        for rel, edge_index in edge_index_dict.items():
            src_t, _, dst_t = rel
            src_idx, dst_idx = edge_index[0], edge_index[1]
            s = self.s_proj[src_t](x_dict[src_t]).view(-1, H, dk)[src_idx]   # [E,H,dk]
            d = self.d_proj[dst_t](x_dict[dst_t]).view(-1, H, dk)[dst_idx]   # [E,H,dk]
            m = self.m_proj[src_t](x_dict[src_t]).view(-1, H, dk)[src_idx]   # [E,H,dk]

            wa = self.w_att[_rk(rel)]                                        # [H,dk,dk]
            wm = self.w_mssg[_rk(rel)]                                       # [H,dk,dk]
            att = torch.einsum("ehi,hij,ehj->eh", s, wa, d) / math.sqrt(dk)  # [E,H]
            alpha = intra_edge_softmax(att)                                  # [E,H]
            msg = torch.einsum("ehi,hij->ehj", m, wm)                        # [E,H,dk]
            contrib = alpha.unsqueeze(-1) * msg                             # [E,H,dk]
            agg[dst_t].index_add_(0, dst_idx, contrib)                      # uniform sum

        out = {}
        for t in x_dict:
            flat = agg[t].reshape(agg[t].size(0), H * dk)
            out[t] = self.out_lin[t](flat) + x_dict[t]                       # Eq. 1 residual
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hgae_conv.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hgae_conv.py tests/test_hgae_conv.py
git commit -m "feat: custom intra-edge attention layer (paper Eq. 1-5)"
```

---

### Task 5: Model — encoder, reparam, per-type decoder, loss (`src/model.py`)

**Files:**
- Modify: `src/model.py` (full rewrite)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `HGAEConv` (Task 4); `build_hetero_data`, `make_featurizers`, `type_targets_from_featurizers`, `HETERO_METADATA` (Tasks 2–3).
- Produces:
  - `HeteroGraphAutoEncoder(metadata, in_dims, type_targets, hidden_dim=64, heads=16, encoder_layers=2, latent_dim=64, decoder_hidden=64, dropout=0.4)` with `forward(x_dict, edge_index_dict) -> (recon, mu, logvar)` where `recon[t] = {"cont": Tensor, "cats": {name: Tensor}}`. Methods `transaction_scores(data, device="cpu") -> Tensor[n_tx]` and `transaction_feature_errors(data, device="cpu") -> dict[str, Tensor[n_tx]]`. Stores `self.type_targets`.
  - `ae_loss(recon, x_dict, type_targets, prevalence) -> (total, parts)` — MSE(continuous) + CE(categorical), prevalence-weighted, **no KL**.
  - `build_model(cfg, metadata, in_dims, type_targets) -> HeteroGraphAutoEncoder`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_model.py`:

```python
import numpy as np
import pandas as pd
import torch

from src.data_prep import (make_featurizers, build_hetero_data,
                           type_targets_from_featurizers, HETERO_METADATA)
from src.model import HeteroGraphAutoEncoder, ae_loss


def _graph():
    n = 12
    df = pd.DataFrame({
        "trans_date_trans_time": ["2020-06-15 14:00:00"] * n,
        "cc_num": [f"c{i%3}" for i in range(n)],
        "merchant": [f"m{i%2}" for i in range(n)],
        "category": ["grocery_pos", "shopping_net"] * (n // 2),
        "amt": np.linspace(5, 500, n), "gender": ["M", "F"] * (n // 2),
        "city_pop": np.linspace(1000, 900000, n),
        "lat": np.linspace(30, 45, n), "long": np.linspace(-120, -75, n),
        "merch_lat": np.linspace(31, 46, n), "merch_long": np.linspace(-119, -74, n),
        "dob": ["1980-01-01"] * n, "is_fraud": ([0] * (n - 1)) + [1],
        "trans_num": [f"t{i}" for i in range(n)],
    })
    feats = make_featurizers()
    for k in feats:
        feats[k].fit(df)
    data, info = build_hetero_data(df, feats)
    tt = type_targets_from_featurizers(feats)
    in_dims = {t: feats[t].dim for t in feats}
    return data, info, tt, in_dims


def _model(tt, in_dims):
    return HeteroGraphAutoEncoder(HETERO_METADATA, in_dims, tt,
                                  hidden_dim=16, heads=2, encoder_layers=2,
                                  latent_dim=16, decoder_hidden=16, dropout=0.0)


def test_forward_recon_shapes():
    data, info, tt, in_dims = _graph()
    model = _model(tt, in_dims)
    recon, mu, logvar = model(data.x_dict, data.edge_index_dict)
    assert recon["transaction"]["cont"].shape == (12, tt["transaction"]["cont"])
    assert recon["merchant"]["cats"]["category"].shape[0] == data["merchant"].x.shape[0]


def test_ae_loss_is_scalar_and_kl_free():
    data, info, tt, in_dims = _graph()
    model = _model(tt, in_dims)
    recon, mu, logvar = model(data.x_dict, data.edge_index_dict)
    prevalence = {t: 1.0 / 3 for t in tt}
    total, parts = ae_loss(recon, data.x_dict, tt, prevalence)
    assert total.ndim == 0
    assert set(parts.keys()) == set(tt.keys())
    # ae_loss must not accept mu/logvar (no KL term)
    import inspect
    assert "mu" not in inspect.signature(ae_loss).parameters
    assert "beta" not in inspect.signature(ae_loss).parameters


def test_transaction_scores_shape_and_positive():
    data, info, tt, in_dims = _graph()
    model = _model(tt, in_dims)
    scores = model.transaction_scores(data)
    assert scores.shape == (12,)
    assert torch.all(scores >= 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL (new signature/attributes not present).

- [ ] **Step 3: Rewrite `src/model.py`**

```python
"""Heterogeneous Graph Auto-Encoder (paper Section 4), faithful implementation.

Encoder : per-type input projection -> stack of custom HGAEConv layers (intra-edge
          attention, Eq. 1-5). After each layer, a VAE-style reparameterization is
          applied per node type (Algorithm 1, line 7).
Decoder : per-type MLP reconstructing each node type's FULL attribute vector -
          continuous features (MSE) and categorical features (cross-entropy),
          Eq. 6-8.
Score   : a transaction's combined reconstruction error (anomaly score).

Two forced deviations from a literal reading (see src/config.py):
  * encoder_layers small (Table 3's 124 oversmooths);
  * standard mu/logvar reparameterization (the literal log(h) formula NaNs).
There is NO KL term (the paper's loss is reconstruction-only).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .hgae_conv import HGAEConv


class HeteroGraphAutoEncoder(nn.Module):
    def __init__(self, metadata, in_dims, type_targets, hidden_dim=64, heads=16,
                 encoder_layers=2, latent_dim=64, decoder_hidden=64, dropout=0.4):
        super().__init__()
        self.metadata = metadata
        self.node_types = list(metadata[0])
        self.type_targets = type_targets
        self.dropout = dropout

        self.in_lin = nn.ModuleDict({t: nn.Linear(in_dims[t], hidden_dim) for t in self.node_types})
        self.convs = nn.ModuleList([HGAEConv(metadata, hidden_dim, heads) for _ in range(encoder_layers)])
        self.fc_mu = nn.ModuleDict({t: nn.Linear(hidden_dim, latent_dim) for t in self.node_types})
        self.fc_logvar = nn.ModuleDict({t: nn.Linear(hidden_dim, latent_dim) for t in self.node_types})

        # per-type decoder: shared trunk -> continuous head + one head per categorical group
        self.dec_trunk = nn.ModuleDict()
        self.dec_cont = nn.ModuleDict()
        self.dec_cat = nn.ModuleDict()
        for t in self.node_types:
            self.dec_trunk[t] = nn.Sequential(
                nn.Linear(latent_dim, decoder_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(decoder_hidden, decoder_hidden), nn.ReLU(), nn.Dropout(dropout),
            )
            self.dec_cont[t] = nn.Linear(decoder_hidden, max(1, type_targets[t]["cont"]))
            self.dec_cat[t] = nn.ModuleDict(
                {name: nn.Linear(decoder_hidden, width)
                 for (name, _start, width) in type_targets[t]["cat_groups"]}
            )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def encode(self, x_dict, edge_index_dict):
        h = {t: F.relu(self.in_lin[t](x)) for t, x in x_dict.items()}
        mu = logvar = None
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {t: F.dropout(F.relu(v), p=self.dropout, training=self.training) for t, v in h.items()}
            mu = {t: self.fc_mu[t](h[t]) for t in h}
            logvar = {t: self.fc_logvar[t](h[t]) for t in h}
            h = {t: self.reparameterize(mu[t], logvar[t]) for t in h}
        return h, mu, logvar

    def forward(self, x_dict, edge_index_dict):
        z, mu, logvar = self.encode(x_dict, edge_index_dict)
        recon = {}
        for t in z:
            trunk = self.dec_trunk[t](z[t])
            recon[t] = {
                "cont": self.dec_cont[t](trunk),
                "cats": {name: head(trunk) for name, head in self.dec_cat[t].items()},
            }
        return recon, mu, logvar

    @torch.no_grad()
    def transaction_scores(self, data, device="cpu") -> torch.Tensor:
        """Per-transaction combined reconstruction error (anomaly score)."""
        self.eval()
        data = data.to(device)
        recon, _, _ = self.forward(data.x_dict, data.edge_index_dict)
        t = "transaction"
        x = data[t].x
        tgt = self.type_targets[t]
        n_cont = tgt["cont"]
        err = ((recon[t]["cont"] - x[:, :n_cont]) ** 2).mean(dim=1)
        for (name, start, width) in tgt["cat_groups"]:
            cls = x[:, start:start + width].argmax(dim=1)
            err = err + F.cross_entropy(recon[t]["cats"][name], cls, reduction="none")
        return err.cpu()

    @torch.no_grad()
    def transaction_feature_errors(self, data, device="cpu") -> dict:
        """Per continuous-feature squared error for transactions (demo 'why')."""
        self.eval()
        data = data.to(device)
        recon, _, _ = self.forward(data.x_dict, data.edge_index_dict)
        t = "transaction"
        x = data[t].x
        n_cont = self.type_targets[t]["cont"]
        sq = (recon[t]["cont"] - x[:, :n_cont]) ** 2
        return {i: sq[:, i].cpu() for i in range(n_cont)}


def ae_loss(recon, x_dict, type_targets, prevalence):
    """Reconstruction loss: MSE(continuous) + CE(categorical), prevalence-weighted.

    NO KL term (the paper's objective is reconstruction-only).
    """
    total = 0.0
    parts = {}
    for t, tgt in type_targets.items():
        x = x_dict[t]
        n_cont = tgt["cont"]
        l = F.mse_loss(recon[t]["cont"], x[:, :n_cont])
        for (name, start, width) in tgt["cat_groups"]:
            cls = x[:, start:start + width].argmax(dim=1)
            l = l + F.cross_entropy(recon[t]["cats"][name], cls)
        parts[t] = l
        total = total + prevalence[t] * l
    return total, parts


def build_model(cfg, metadata, in_dims, type_targets) -> HeteroGraphAutoEncoder:
    return HeteroGraphAutoEncoder(
        metadata=metadata, in_dims=in_dims, type_targets=type_targets,
        hidden_dim=cfg.hidden_dim, heads=cfg.heads, encoder_layers=cfg.encoder_layers,
        latent_dim=cfg.latent_dim, decoder_hidden=cfg.decoder_hidden, dropout=cfg.dropout,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: faithful encoder/decoder + MSE+CE loss, no KL"
```

---

### Task 6: μ+2σ threshold (`src/evaluate.py`)

**Files:**
- Modify: `src/evaluate.py` (add `mu_2sigma_threshold`; keep everything else)
- Test: `tests/test_threshold.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `mu_2sigma_threshold(genuine_scores) -> float` = `mean + 2*std` of the genuine reconstruction errors. `search_threshold` and `compute_metrics` unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_threshold.py`:

```python
import numpy as np

from src.evaluate import mu_2sigma_threshold


def test_mu_2sigma_value():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(mu_2sigma_threshold(s) - (s.mean() + 2 * s.std())) < 1e-9


def test_mu_2sigma_covers_about_95_percent():
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 1.0, size=100_000)
    thr = mu_2sigma_threshold(s)
    frac_below = (s < thr).mean()
    assert 0.95 < frac_below < 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_threshold.py -v`
Expected: FAIL (`mu_2sigma_threshold` not defined).

- [ ] **Step 3: Implement in `src/evaluate.py`**

Add after `search_threshold`:

```python
def mu_2sigma_threshold(genuine_scores) -> float:
    """Paper Eq. 9: Threshold = mu + 2*sigma of genuine reconstruction errors."""
    s = np.asarray(genuine_scores, dtype=float)
    return float(s.mean() + 2.0 * s.std())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_threshold.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_threshold.py
git commit -m "feat: mu+2sigma threshold rule (paper Eq. 9)"
```

---

### Task 7: Training pipeline (`src/train.py`)

**Files:**
- Modify: `src/train.py` (full rewrite of `run_training`; keep `_split_train`, `ARTIFACT_PATH_NAME`)
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces: `run_training(cfg) -> dict`. Saves `outputs/artifacts.pt` with keys: `model_state, model_kwargs (metadata, in_dims, type_targets, hidden_dim, heads, encoder_layers, latent_dim, decoder_hidden, dropout), featurizers (dict kind->asdict), reference_store, demo_aux, prevalence, threshold_mu2sigma, threshold_f1, metrics, metrics_f1, feature_names (per type), is_sample_data`. Uses `threshold_mu2sigma` for the reported metrics.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_train_smoke.py`:

```python
import numpy as np
import pandas as pd
import torch

from src.data_prep import (make_featurizers, build_hetero_data,
                           type_targets_from_featurizers, HETERO_METADATA)
from src.model import build_model, ae_loss
from src.evaluate import mu_2sigma_threshold


class _Cfg:
    hidden_dim = 16; heads = 2; encoder_layers = 2; latent_dim = 16
    decoder_hidden = 16; dropout = 0.0


def _df(n, fraud_tail=0):
    y = [0] * (n - fraud_tail) + [1] * fraud_tail
    return pd.DataFrame({
        "trans_date_trans_time": ["2020-06-15 14:00:00"] * n,
        "cc_num": [f"c{i%4}" for i in range(n)],
        "merchant": [f"m{i%3}" for i in range(n)],
        "category": (["grocery_pos", "shopping_net", "gas_transport"] * n)[:n],
        "amt": np.linspace(5, 800, n), "gender": (["M", "F"] * n)[:n],
        "city_pop": np.linspace(1000, 900000, n),
        "lat": np.linspace(30, 45, n), "long": np.linspace(-120, -75, n),
        "merch_lat": np.linspace(31, 46, n), "merch_long": np.linspace(-119, -74, n),
        "dob": ["1980-01-01"] * n, "is_fraud": y,
        "trans_num": [f"t{i}" for i in range(n)],
    })


def test_one_training_step_and_scoring():
    df = _df(40)
    feats = make_featurizers()
    for k in feats:
        feats[k].fit(df)
    data, info = build_hetero_data(df, feats)
    tt = type_targets_from_featurizers(feats)
    in_dims = {t: feats[t].dim for t in feats}
    model = build_model(_Cfg(), HETERO_METADATA, in_dims, tt)

    counts = {t: data[t].x.size(0) for t in in_dims}
    tot = sum(counts.values())
    prevalence = {t: counts[t] / tot for t in counts}

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    recon, mu, logvar = model(data.x_dict, data.edge_index_dict)
    loss, parts = ae_loss(recon, data.x_dict, tt, prevalence)
    loss.backward(); opt.step()
    assert torch.isfinite(loss)

    scores = model.transaction_scores(data).numpy()
    thr = mu_2sigma_threshold(scores)
    assert np.isfinite(thr)
    assert scores.shape == (40,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_smoke.py -v`
Expected: FAIL until `build_model`/`ae_loss` have the new signatures (they do after Task 5) — if Task 5 done, this may already PASS. If it passes, still proceed to rewrite `run_training` (its own behavior is exercised by Task 8's loaded artifacts). Confirm by running the full suite in Step 4.

- [ ] **Step 3: Rewrite `run_training` in `src/train.py`**

Replace the body after `_split_train` with:

```python
from dataclasses import asdict

from . import data_prep, evaluate
from .config import Config
from .model import ae_loss, build_model
from .utils import pick_device, set_seed


def run_training(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    train_path, test_path, is_sample = data_prep.resolve_data_paths(cfg)
    print(f"[data] train={train_path.name} test={test_path.name} (synthetic={is_sample})")
    train_raw = data_prep.load_raw(train_path, cfg.train_subsample, cfg.keep_all_fraud, cfg.seed)
    test_raw = data_prep.load_raw(test_path, cfg.test_subsample, cfg.keep_all_fraud, cfg.seed)
    print(f"[data] train rows={len(train_raw)} fraud={int(train_raw.is_fraud.sum())} | "
          f"test rows={len(test_raw)} fraud={int(test_raw.is_fraud.sum())}")

    train_gen, val_df = _split_train(train_raw, cfg.val_fraction, cfg.seed)

    feats = data_prep.make_featurizers()
    for k in feats:
        feats[k].fit(train_gen)                       # genuine-only fit, no leakage
    print(f"[feat] dims: " + ", ".join(f"{t}={feats[t].dim}" for t in feats))

    train_data, _ = data_prep.build_hetero_data(train_gen, feats)
    val_data, val_info = data_prep.build_hetero_data(val_df, feats)
    test_data, test_info = data_prep.build_hetero_data(test_raw, feats)

    metadata = data_prep.HETERO_METADATA
    in_dims = {t: feats[t].dim for t in feats}
    type_targets = data_prep.type_targets_from_featurizers(feats)

    import torch
    train_data = train_data.to(device)
    counts = {t: train_data[t].x.size(0) for t in in_dims}
    total_nodes = sum(counts.values())
    prevalence = {t: counts[t] / total_nodes for t in counts}   # Eq. 7 prevalence weights

    model = build_model(cfg, metadata, in_dims, type_targets).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    val_labels = val_info["labels"]
    history = {"train_loss": [], "val_loss": [], "val_auc_pr": []}
    best_auc, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        recon, mu, logvar = model(train_data.x_dict, train_data.edge_index_dict)
        loss, parts = ae_loss(recon, train_data.x_dict, type_targets, prevalence)
        loss.backward()
        opt.step()

        val_scores = model.transaction_scores(val_data, device).numpy()
        gen = val_labels == 0
        try:
            from sklearn.metrics import average_precision_score
            val_auc = float(average_precision_score(val_labels, val_scores))
        except Exception:
            val_auc = float("nan")
        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_scores[gen].mean()))
        history["val_auc_pr"].append(val_auc)

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"[epoch {epoch:3d}] loss={loss.item():.5f} val_auc_pr={val_auc:.4f}")
        if patience >= cfg.early_stop_patience:
            print(f"[early stop] no val improvement for {cfg.early_stop_patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- thresholds: mu+2sigma (decision rule) + F1-sweep (comparison) ----- #
    val_scores = model.transaction_scores(val_data, device).numpy()
    genuine_val = val_scores[val_labels == 0]
    threshold_mu2sigma = evaluate.mu_2sigma_threshold(genuine_val)
    threshold_f1, val_f1 = evaluate.search_threshold(val_scores, val_labels)

    test_scores = model.transaction_scores(test_data, device).numpy()
    test_labels = test_info["labels"]
    metrics = evaluate.compute_metrics(test_scores, test_labels, threshold_mu2sigma)
    metrics_f1 = evaluate.compute_metrics(test_scores, test_labels, threshold_f1)
    print(f"[eval] mu+2sigma thr={threshold_mu2sigma:.6f}  "
          f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
          f"AUC-PR={metrics['auc_pr']:.3f} ROC-AUC={metrics['roc_auc']:.3f}")
    print(f"[eval] F1-sweep thr={threshold_f1:.6f} (val F1={val_f1:.3f})  "
          f"test F1={metrics_f1['f1']:.3f}")

    plot_paths = evaluate.make_all_plots(test_scores, test_labels, history, cfg.output_dir / "plots")
    print(f"[plots] wrote {len(plot_paths)} figures")

    ref_store = data_prep.compute_reference_store(train_gen, feats)
    demo_aux = data_prep.build_demo_aux(train_raw, test_raw, seed=cfg.seed)

    model_kwargs = dict(metadata=metadata, in_dims=in_dims, type_targets=type_targets,
                        hidden_dim=cfg.hidden_dim, heads=cfg.heads,
                        encoder_layers=cfg.encoder_layers, latent_dim=cfg.latent_dim,
                        decoder_hidden=cfg.decoder_hidden, dropout=cfg.dropout)
    artifact = {
        "model_state": model.state_dict(),
        "model_kwargs": model_kwargs,
        "featurizers": {k: asdict(f) for k, f in feats.items()},
        "reference_store": ref_store,
        "demo_aux": demo_aux,
        "prevalence": prevalence,
        "threshold_mu2sigma": float(threshold_mu2sigma),
        "threshold_f1": float(threshold_f1),
        "metrics": metrics,
        "metrics_f1": metrics_f1,
        "feature_names": {t: feats[t].feature_names for t in feats},
        "is_sample_data": is_sample,
    }
    out_path = cfg.output_dir / ARTIFACT_PATH_NAME
    torch.save(artifact, out_path)
    import json
    with open(cfg.output_dir / "metrics.json", "w") as f:
        json.dump({"test_mu2sigma": metrics, "test_f1sweep": metrics_f1,
                   "threshold_mu2sigma": threshold_mu2sigma, "threshold_f1": threshold_f1,
                   "best_val_auc_pr": best_auc, "is_sample_data": is_sample}, f, indent=2)
    print(f"[save] artifacts -> {out_path}")
    return {"metrics": metrics, "threshold": threshold_mu2sigma, "artifact_path": str(out_path)}
```

Remove the old imports at the top of the file that are now duplicated inside/above (ensure a single import block; keep `numpy`, `pandas`, `time` only if still referenced — `time` is no longer used, remove it).

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -v`
Expected: all tests PASS. `test_train_smoke.py` confirms build_model/ae_loss/scoring integrate.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train_smoke.py
git commit -m "feat: training pipeline with mu+2sigma + F1-sweep thresholds"
```

---

### Task 8: Scorer (`src/scorer.py`)

**Files:**
- Modify: `src/scorer.py` (full rewrite)
- Test: `tests/test_scorer.py`

**Interfaces:**
- Consumes: Task 7 artifacts, `TypeFeaturizer`, `HeteroGraphAutoEncoder`.
- Produces: `FraudScorer` with `classmethod load(artifact_path, device="cpu")`, `score_transaction(record, top_k=5) -> dict` (keys `reconstruction_error, threshold, ratio, is_fraud, verdict, customer_known, merchant_known, top_features`), and `known_customers/known_merchants(limit=None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scorer.py`:

```python
import numpy as np
import pandas as pd
import torch

from src.data_prep import (make_featurizers, build_hetero_data,
                           type_targets_from_featurizers, HETERO_METADATA,
                           compute_reference_store)
from src.model import build_model
from src.scorer import FraudScorer
from dataclasses import asdict


class _Cfg:
    hidden_dim = 16; heads = 2; encoder_layers = 2; latent_dim = 16
    decoder_hidden = 16; dropout = 0.0


def _df(n=40):
    return pd.DataFrame({
        "trans_date_trans_time": ["2020-06-15 14:00:00"] * n,
        "cc_num": [f"c{i%4}" for i in range(n)],
        "merchant": [f"m{i%3}" for i in range(n)],
        "category": (["grocery_pos", "shopping_net", "gas_transport"] * n)[:n],
        "amt": np.linspace(5, 800, n), "gender": (["M", "F"] * n)[:n],
        "city_pop": np.linspace(1000, 900000, n),
        "lat": np.linspace(30, 45, n), "long": np.linspace(-120, -75, n),
        "merch_lat": np.linspace(31, 46, n), "merch_long": np.linspace(-119, -74, n),
        "dob": ["1980-01-01"] * n, "is_fraud": [0] * n,
        "trans_num": [f"t{i}" for i in range(n)],
    })


def _artifact(tmp_path):
    df = _df()
    feats = make_featurizers()
    for k in feats:
        feats[k].fit(df)
    tt = type_targets_from_featurizers(feats)
    in_dims = {t: feats[t].dim for t in feats}
    model = build_model(_Cfg(), HETERO_METADATA, in_dims, tt)
    art = {
        "model_state": model.state_dict(),
        "model_kwargs": dict(metadata=HETERO_METADATA, in_dims=in_dims, type_targets=tt,
                             hidden_dim=16, heads=2, encoder_layers=2, latent_dim=16,
                             decoder_hidden=16, dropout=0.0),
        "featurizers": {k: asdict(f) for k, f in feats.items()},
        "reference_store": compute_reference_store(df, feats),
        "demo_aux": {}, "threshold_mu2sigma": 0.5, "threshold_f1": 0.5,
        "metrics": {}, "metrics_f1": {},
        "feature_names": {t: feats[t].feature_names for t in feats},
        "is_sample_data": True,
    }
    p = tmp_path / "artifacts.pt"
    torch.save(art, p)
    return p, df


def test_score_known_and_coldstart(tmp_path):
    p, df = _artifact(tmp_path)
    scorer = FraudScorer.load(str(p))
    rec_known = df.iloc[0].to_dict()
    out = scorer.score_transaction(rec_known)
    assert out["customer_known"] is True
    assert out["verdict"] in ("FRAUD", "NON-FRAUD")
    assert isinstance(out["reconstruction_error"], float)
    assert len(out["top_features"]) >= 1

    rec_new = dict(rec_known, cc_num="ZZZ_new", merchant="ZZZ_newmerch")
    out2 = scorer.score_transaction(rec_new)
    assert out2["customer_known"] is False
    assert out2["merchant_known"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scorer.py -v`
Expected: FAIL (old `FraudScorer.load` expects `featurizer`/`model_kwargs` in the old format).

- [ ] **Step 3: Rewrite `src/scorer.py`**

```python
"""Real-time fraud scorer used by the Streamlit demo.

Loads the artifacts saved by training and scores a SINGLE transaction with no
retraining. It builds a 3-node heterogeneous graph (the transaction + its
customer + its merchant), each node featurized by its OWN type featurizer, and
compares the transaction's reconstruction error to the mu+2sigma threshold.

* known customer/merchant -> use the stored genuine feature vector
* new customer/merchant   -> cold-start from the incoming record's own fields
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_prep import TypeFeaturizer, HETERO_METADATA
from .model import HeteroGraphAutoEncoder


class FraudScorer:
    def __init__(self, model, feats, reference_store, threshold, feature_names,
                 metrics, device="cpu", demo_aux=None, is_sample=False,
                 metrics_f1=None, threshold_f1=None):
        self.model = model
        self.feats = feats                      # dict type -> TypeFeaturizer
        self.ref = reference_store
        self.threshold = float(threshold)
        self.threshold_f1 = float(threshold_f1) if threshold_f1 is not None else None
        self.feature_names = feature_names      # dict type -> list
        self.metrics = metrics or {}
        self.metrics_f1 = metrics_f1 or {}
        self.device = device
        self.demo_aux = demo_aux or {}
        self.is_sample = is_sample
        self.txn_cont_names = feats["transaction"].cont_cols
        self.model.eval()

    @classmethod
    def load(cls, artifact_path: str | Path, device: str = "cpu") -> "FraudScorer":
        import torch
        art = torch.load(artifact_path, map_location=device, weights_only=False)
        feats = {k: TypeFeaturizer(**d) for k, d in art["featurizers"].items()}
        model = HeteroGraphAutoEncoder(**art["model_kwargs"])
        model.load_state_dict(art["model_state"])
        model.to(device).eval()
        return cls(model, feats, art["reference_store"], art["threshold_mu2sigma"],
                   art["feature_names"], art.get("metrics", {}), device,
                   demo_aux=art.get("demo_aux", {}), is_sample=art.get("is_sample_data", False),
                   metrics_f1=art.get("metrics_f1", {}), threshold_f1=art.get("threshold_f1"))

    def known_customers(self, limit=None):
        ids = list(self.ref.get("customer", {}).keys())
        return ids[:limit] if limit else ids

    def known_merchants(self, limit=None):
        ids = list(self.ref.get("merchant", {}).keys())
        return ids[:limit] if limit else ids

    def _profile(self, kind, key, fallback):
        table = self.ref.get(kind, {})
        if key in table:
            return np.asarray(table[key], dtype=np.float32), True
        return fallback.astype(np.float32), False

    def _build_single_graph(self, x_txn, cust_vec, merch_vec):
        import torch
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["transaction"].x = torch.tensor(x_txn[None, :], dtype=torch.float32)
        data["customer"].x = torch.tensor(cust_vec[None, :], dtype=torch.float32)
        data["merchant"].x = torch.tensor(merch_vec[None, :], dtype=torch.float32)
        e = torch.tensor([[0], [0]], dtype=torch.long)
        data["customer", "makes", "transaction"].edge_index = e.clone()
        data["transaction", "rev_makes", "customer"].edge_index = e.clone()
        data["merchant", "sells", "transaction"].edge_index = e.clone()
        data["transaction", "rev_sells", "merchant"].edge_index = e.clone()
        return data

    def score_transaction(self, record: dict, top_k: int = 5) -> dict:
        x_txn = self.feats["transaction"].transform_one(record)
        cust_cold = self.feats["customer"].transform_one(record)
        merch_cold = self.feats["merchant"].transform_one(record)
        cust_vec, cust_known = self._profile("customer", str(record.get("cc_num", "")), cust_cold)
        merch_vec, merch_known = self._profile("merchant", str(record.get("merchant", "")), merch_cold)

        data = self._build_single_graph(x_txn, cust_vec, merch_vec)
        error = float(self.model.transaction_scores(data, self.device)[0])
        feat_err = self.model.transaction_feature_errors(data, self.device)
        per_feat = np.array([float(feat_err[i][0]) for i in range(len(self.txn_cont_names))])

        is_fraud = error >= self.threshold
        order = np.argsort(per_feat)[::-1][:top_k]
        top_features = [
            {"feature": self.txn_cont_names[i], "error": float(per_feat[i]),
             "value": float(x_txn[i])}
            for i in order
        ]
        return {
            "reconstruction_error": error,
            "threshold": self.threshold,
            "ratio": error / self.threshold if self.threshold > 0 else float("inf"),
            "is_fraud": bool(is_fraud),
            "verdict": "FRAUD" if is_fraud else "NON-FRAUD",
            "customer_known": cust_known,
            "merchant_known": merch_known,
            "top_features": top_features,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scorer.py -v`
Expected: PASS (1 test, 2 assertions blocks).

- [ ] **Step 5: Commit**

```bash
git add src/scorer.py tests/test_scorer.py
git commit -m "feat: per-type scorer with combined recon error + cold-start"
```

---

### Task 9: Demo app + docs (`app.py`, `CLAUDE.md`)

**Files:**
- Modify: `app.py` (only the sidebar metrics + result display that referenced old fields)
- Modify: `CLAUDE.md` (rewrite conventions + deviations)
- Test: manual (Streamlit); plus full pytest suite must still pass.

**Interfaces:**
- Consumes: `FraudScorer` (Task 8). `scorer.metrics` (mu+2σ operating point), `scorer.metrics_f1`, `scorer.threshold`.

- [ ] **Step 1: Update `app.py` sidebar + caption**

The scorer no longer exposes `metrics_balanced`; it exposes `metrics` (mu+2σ) and `metrics_f1`. In `app.py`, replace the sidebar "Model performance" block (currently reading `scorer.metrics_balanced`) with:

```python
    with st.sidebar:
        st.header("Model performance")
        m = scorer.metrics
        if m:
            st.caption("Full test @ threshold = μ+2σ (paper Eq. 9)")
            st.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
            st.metric("AUC-PR", f"{m.get('auc_pr', float('nan')):.3f}")
            st.metric("F1", f"{m.get('f1', float('nan')):.3f}")
            st.metric("Precision / Recall",
                      f"{m.get('precision', 0):.2f} / {m.get('recall', 0):.2f}")
        if scorer.metrics_f1:
            st.caption(f"(F1-sweep F1 = {scorer.metrics_f1.get('f1', float('nan')):.3f})")
        st.caption(f"Decision threshold = `{scorer.threshold:.6f}`")
        st.caption(f"Known customers: {len(known_customers)} · merchants: {len(known_merchants)}")
        if scorer.is_sample:
            st.warning("Model trained on **synthetic sample data**. Put the real "
                       "Kaggle CSVs in `data/` or `archive/` and rerun `python main.py`.")
```

Update the top caption to the correct attribution (remove the arXiv/Singh mislabel):

```python
    st.caption("Demo of Majumder et al., *Heterogeneous Graph Auto-Encoder for "
               "Credit Card Fraud Detection* (IJCA 32(2), 2025). The model learns "
               "to reconstruct **genuine** transactions; a high reconstruction "
               "error ⇒ fraud.")
```

The result display (`_show_result`) already consumes `res["top_features"]` with `feature/error/value` — unchanged.

- [ ] **Step 2: Verify the app imports cleanly**

Run: `python -c "import ast; ast.parse(open('app.py').read()); print('app.py parses')"`
Expected: `app.py parses`.

- [ ] **Step 3: Rewrite `CLAUDE.md` conventions + deviations**

Replace the "Project-specific conventions" and "Deviations from the paper" sections with:

```markdown
## Project-specific conventions

- **Faithful to the paper.** This code implements Majumder et al. (IJCA 32(2)
  2025) as described: per-type node attributes, custom intra-edge attention
  (softmax across heads, uniform neighbour sum), full attribute reconstruction
  (MSE for continuous + cross-entropy for categorical, prevalence-weighted),
  reparameterization with NO KL term, and a μ+2σ decision threshold.
- **Per-type attributes** (`data_prep.py`): customer = demographics, merchant =
  category + location, transaction = amount/time/distance. Each node type has its
  own featurizer and its own attribute dimensionality (this is why the encoder's
  type-specific projections matter). Continuous columns come FIRST in every
  vector, then one-hot categorical blocks.
- **Custom attention** (`hgae_conv.py`): `HGAEConv` implements paper Eq. 1-5.
  Softmax is over the H heads of each edge; neighbours are summed uniformly. Do
  NOT swap this for PyG `HGTConv` (which normalizes across neighbours instead).
- **Threshold = μ+2σ** on held-out genuine reconstruction errors (paper Eq. 9);
  the F1-sweep operating point is also computed and logged for comparison.

## Deviations from the paper (the only two — both forced)

- Table 3's "124 encoder layers" → `encoder_layers=2` (124 oversmooths
  catastrophically; embeddings collapse).
- The literal reparameterization `mean(h)+ε·exp(½·log(h))` takes `log()` of raw
  activations (NaN) → standard `mu`/`logvar` heads instead.

Everything else matches Table 3: heads=16, dropout=0.4, regularization
(weight_decay)=0.01, no KL, no latent bottleneck (latent=hidden=64).
```

Also fix the first line of CLAUDE.md that cites "arXiv:2410.08121" and "Singh et al." → "(Majumder et al., IJCA 32(2) 2025)".

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py CLAUDE.md
git commit -m "feat: align demo + CLAUDE.md with faithful implementation"
```

---

### Task 10: End-to-end run on synthetic data (verification)

**Files:** none (verification only).

- [ ] **Step 1: Generate synthetic data**

Run: `python make_sample_data.py`
Expected: creates `data/sample/fraudTrain.csv` and `fraudTest.csv`. If it errors on a missing column needed by the featurizers, extend `make_sample_data.py` to emit it (it already emits the full Sparkov schema — this should not happen).

- [ ] **Step 2: Train a short run**

Run: `python main.py --epochs 10 --train-subsample 20000`
Expected: completes without error; prints `[eval] mu+2sigma thr=...` and F1-sweep line; writes `outputs/artifacts.pt`, `outputs/metrics.json`, and 5 plots under `outputs/plots/`.

- [ ] **Step 3: Smoke-test the scorer against the real artifact**

Run:
```bash
python -c "from src.scorer import FraudScorer; s=FraudScorer.load('outputs/artifacts.pt'); import json; print(json.dumps(s.score_transaction({'cc_num':'0','merchant':'x','category':'shopping_net','amt':999.0,'gender':'M','city_pop':50000,'lat':38.0,'long':-95.0,'merch_lat':38.0,'merch_long':-95.0,'dob':'1980-01-01','trans_date_trans_time':'2020-06-15 03:00:00'}), default=str))"
```
Expected: prints a JSON verdict dict with `reconstruction_error`, `verdict`, `top_features`.

- [ ] **Step 4: Commit any fixes surfaced by the run**

```bash
git add -A
git commit -m "fix: issues surfaced by end-to-end synthetic run"
```
(Skip if nothing changed.)

---

## Self-Review

**Spec coverage:**
- Heterogeneous per-type attributes → Task 2 + 3. ✓
- Custom intra-edge attention (Eq. 1-5) → Task 4. ✓
- Reparam per layer, no bottleneck, stable heads → Task 5. ✓
- Full attribute reconstruction MSE+CE, prevalence-weighted, no KL → Task 5. ✓
- μ+2σ threshold + F1-sweep comparison → Task 6 + 7. ✓
- Faithful hyperparameters (16 heads, dropout 0.4, wd 0.01, β=0) → Task 1. ✓
- Forced deviations documented (depth, reparam) → Task 1 config docstring + CLAUDE.md Task 9. ✓
- Scorer per-type + cold-start → Task 8. ✓
- Demo + CLAUDE.md updated, citation fixed → Task 9. ✓
- Subsample default kept, `--full` intact → unchanged `main.py`/`config.py`; verified Task 10. ✓
- Edge naming + Refunds-not-modeled note → Task 3 (edges) + design doc. ✓

**Type consistency:** `make_featurizers`, `type_targets_from_featurizers`, `build_hetero_data(df, feats)`, `HETERO_METADATA`, `build_model(cfg, metadata, in_dims, type_targets)`, `ae_loss(recon, x_dict, type_targets, prevalence)`, `transaction_scores`, `transaction_feature_errors`, `mu_2sigma_threshold`, artifact keys (`featurizers`, `threshold_mu2sigma`, `threshold_f1`, `metrics`, `metrics_f1`, `feature_names`) — used identically across Tasks 2–9. ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code. ✓

## Notes / known risks (from the spec)
- Fidelity may lower detection quality vs. the tuned version (uniform-neighbour sum, no KL, no bottleneck). Expected; documented, not "fixed".
- `torch`/`torch_geometric`/`pytest` are not yet installed (Task 1 installs them).
- Full-graph μ+2σ on high-degree nodes: fine on the subsample default; `--full` on CPU may be slow/memory-heavy (documented limitation, not addressed).
