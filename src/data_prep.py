"""Data loading, feature engineering and heterogeneous-graph construction.

Design goals
------------
* Every transaction feature is computable from a SINGLE raw record, so the exact
  same featuriser is reused at inference time (the Streamlit demo).
* Customer / merchant node features = the mean of the feature vectors of the
  transactions attached to them. This keeps all node types in the same feature
  space (no extra scalers), is inductive, and makes "cold-start" trivial: a brand
  new customer/merchant just inherits the feature vector of the incoming
  transaction.

Heterogeneous graph (paper Section 3): three node types, each with its own
attribute set (per-type featurizers):
    customer  (cc_num) -> TypeFeaturizer("customer", ...)
    merchant  (merchant) -> TypeFeaturizer("merchant", ...)
    transaction (trans_num) -> TypeFeaturizer("transaction", ...)
with bidirectional edges  customer <-> transaction <-> merchant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import os
import random

import numpy as np
import pandas as pd

# ----- columns the featuriser depends on (must be present in any input record) -----
REQUIRED_COLS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "city_pop", "lat", "long", "merch_lat", "merch_long", "dob",
]

# ----- per-type attribute definitions (paper: heterogeneous attribute sets) -----
CUST_CONT = ["age", "log_city_pop", "home_lat", "home_long"]
CUST_CAT = ["gender"]
MERC_CONT = ["merch_lat_f", "merch_long_f"]
MERC_CAT = ["category"]
TXN_CONT = ["log_amt", "log_distance", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TXN_CAT: list[str] = []


# --------------------------------------------------------------------------- #
# path resolution + loading
# --------------------------------------------------------------------------- #
def resolve_data_paths(cfg) -> tuple[Path, Path]:
    """Locate the real Kaggle files (data/ or archive/). Raises if not found."""
    project_root = cfg.train_csv.parent.parent
    candidates = [
        (cfg.train_csv, cfg.test_csv),
        (project_root / "archive" / "fraudTrain.csv", project_root / "archive" / "fraudTest.csv"),
        (cfg.train_csv.parent / "archive" / "fraudTrain.csv",
         cfg.train_csv.parent / "archive" / "fraudTest.csv"),
    ]
    for train_p, test_p in candidates:
        if train_p.exists() and test_p.exists():
            return train_p, test_p
    raise FileNotFoundError(
        "No dataset found. Download the Kaggle 'kartik2112/fraud-detection' dataset "
        "and place fraudTrain.csv & fraudTest.csv in data/ or archive/."
    )


def load_raw(path: Path, subsample: int | None, keep_all_fraud: bool, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    # first unnamed column is a row index in the Kaggle files
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=df.columns[0])
    missing = [c for c in REQUIRED_COLS + ["is_fraud", "trans_num"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing expected columns: {missing}")

    if subsample is not None and subsample < len(df):
        rng = np.random.default_rng(seed)
        if keep_all_fraud:
            fraud = df[df["is_fraud"] == 1]
            genuine = df[df["is_fraud"] == 0]
            n_gen = max(0, subsample - len(fraud))
            gen_idx = rng.choice(genuine.index.values, size=min(n_gen, len(genuine)), replace=False)
            df = pd.concat([fraud, genuine.loc[gen_idx]]).sort_index()
        else:
            idx = rng.choice(df.index.values, size=subsample, replace=False)
            df = df.loc[idx].sort_index()
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# feature engineering
# --------------------------------------------------------------------------- #
def _haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


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
        elif self.kind == "transaction":
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
        else:
            raise ValueError(f"unknown TypeFeaturizer kind: {self.kind!r}")
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
        if self.mean_ is None:
            raise RuntimeError("TypeFeaturizer.dim read before fit(); call fit() first")
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


# --------------------------------------------------------------------------- #
# helper for graph construction & demo
# --------------------------------------------------------------------------- #
def _mean_by_group(X: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    F = X.shape[1]
    sums = np.zeros((n_groups, F), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.float64)
    np.add.at(sums, group_idx, X)
    np.add.at(counts, group_idx, 1.0)
    counts[counts == 0] = 1.0
    return (sums / counts[:, None]).astype(np.float32)


def build_demo_aux(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   n_customers: int = 300, n_merchants: int = 200,
                   n_examples: int = 25, seed: int = 42) -> dict:
    """Auxiliary data so the Streamlit demo can autofill realistic records.

    * customer_attrs : cc_num -> static attributes (home coords, demographics)
    * merchant_attrs : merchant -> category + merchant coords
    * examples       : a few raw genuine/fraud rows from the test set to load & tweak
    """
    rng = np.random.default_rng(seed)

    cust_cols = ["lat", "long", "city_pop", "gender", "dob", "first", "last", "city", "state", "job"]
    cust = train_df.drop_duplicates("cc_num")
    customer_attrs = {}
    for _, row in cust.head(n_customers).iterrows():
        customer_attrs[str(row["cc_num"])] = {c: _py(row[c]) for c in cust_cols if c in row}

    merch_group = train_df.groupby("merchant")
    merchant_attrs = {}
    for name, g in list(merch_group)[:n_merchants]:
        merchant_attrs[str(name)] = {
            "category": str(g["category"].mode().iloc[0]),
            "merch_lat": float(g["merch_lat"].mean()),
            "merch_long": float(g["merch_long"].mean()),
        }

    ex_cols = REQUIRED_COLS + ["cc_num", "merchant", "is_fraud", "trans_num"]
    ex_cols = [c for c in dict.fromkeys(ex_cols) if c in test_df.columns]

    def _sample(df, n):
        if len(df) == 0:
            return []
        idx = rng.choice(df.index.values, size=min(n, len(df)), replace=False)
        return [{c: _py(v) for c, v in r.items()} for r in df.loc[idx, ex_cols].to_dict("records")]

    examples = {
        "genuine": _sample(test_df[test_df["is_fraud"] == 0], n_examples),
        "fraud": _sample(test_df[test_df["is_fraud"] == 1], n_examples),
    }
    return {"customer_attrs": customer_attrs, "merchant_attrs": merchant_attrs,
            "examples": examples, "categories": sorted(train_df["category"].astype(str).unique().tolist())}


def _py(v):
    """Convert numpy scalars to plain Python for clean pickling/JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def build_viz_graph(df: pd.DataFrame, max_nodes: int = 100, seed: int = 42,
                    fraud_frac: float = 0.3) -> dict:
    """Sampled heterogeneous subgraph for the demo's interactive 3D view.

    Samples up to `max_nodes` transactions with fraud OVER-sampled to roughly
    `fraud_frac` of the total (real fraud is ~0.5%, so a natural sample would
    show almost none), then adds the customers and merchants they touch. Returns
    a JSON-serializable dict::

        {"nodes": [{"id","type","is_fraud"?,"raw":{...}}, ...],
         "links": [{"source","target","relation"}, ...]}

    `raw` holds human-readable fields shown in the hover tooltip. Node ids are
    namespaced by type ("txn:"/"cust:"/"merch:") so they never collide.
    """
    if len(df) == 0:
        return {"nodes": [], "links": []}
    rng = np.random.default_rng(seed)
    n = min(max_nodes, len(df))

    fraud_idx = df.index.values[df["is_fraud"].to_numpy() == 1]
    gen_idx = df.index.values[df["is_fraud"].to_numpy() == 0]
    n_fraud = min(len(fraud_idx), int(round(n * fraud_frac)))
    n_gen = min(len(gen_idx), n - n_fraud)
    n_fraud = min(len(fraud_idx), n - n_gen)          # backfill if genuine is short
    pick = np.concatenate([
        rng.choice(fraud_idx, size=n_fraud, replace=False),
        rng.choice(gen_idx, size=n_gen, replace=False),
    ])
    rng.shuffle(pick)
    sample = df.loc[pick]

    dist = _haversine_km(
        pd.to_numeric(sample["lat"], errors="coerce").fillna(0.0).values,
        pd.to_numeric(sample["long"], errors="coerce").fillna(0.0).values,
        pd.to_numeric(sample["merch_lat"], errors="coerce").fillna(0.0).values,
        pd.to_numeric(sample["merch_long"], errors="coerce").fillna(0.0).values,
    )

    nodes: list[dict] = []
    links: list[dict] = []
    seen_cust: set[str] = set()
    seen_merch: set[str] = set()

    for pos, (_, row) in enumerate(sample.iterrows()):
        cc = str(row["cc_num"])
        merch = str(row["merchant"])
        txn_id = f"txn:{row['trans_num']}"
        cust_id = f"cust:{cc}"
        merch_id = f"merch:{merch}"

        nodes.append({
            "id": txn_id, "type": "transaction", "is_fraud": int(row["is_fraud"]),
            "raw": {
                "amt": _py(row.get("amt")),
                "category": _py(row.get("category")),
                "datetime": _py(row.get("trans_date_trans_time")),
                "distance_km": round(float(dist[pos]), 2),
                "is_fraud": int(row["is_fraud"]),
                "trans_num": _py(row.get("trans_num")),
            },
            # coordinates for the demo's map view (home -> merchant arc)
            "geo": {
                "home_lat": float(pd.to_numeric(row.get("lat"), errors="coerce") or 0.0),
                "home_lon": float(pd.to_numeric(row.get("long"), errors="coerce") or 0.0),
                "merch_lat": float(pd.to_numeric(row.get("merch_lat"), errors="coerce") or 0.0),
                "merch_lon": float(pd.to_numeric(row.get("merch_long"), errors="coerce") or 0.0),
            },
        })
        if cust_id not in seen_cust:
            seen_cust.add(cust_id)
            nodes.append({
                "id": cust_id, "type": "customer",
                "raw": {c: _py(row.get(c)) for c in
                        ("cc_num", "gender", "dob", "city", "state", "city_pop", "job")
                        if c in row},
            })
        if merch_id not in seen_merch:
            seen_merch.add(merch_id)
            nodes.append({
                "id": merch_id, "type": "merchant",
                "raw": {
                    "merchant": merch,
                    "category": _py(row.get("category")),
                    "merch_lat": _py(row.get("merch_lat")),
                    "merch_long": _py(row.get("merch_long")),
                },
            })
        links.append({"source": cust_id, "target": txn_id, "relation": "makes"})
        links.append({"source": merch_id, "target": txn_id, "relation": "sells"})

    return {"nodes": nodes, "links": links}


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
    df_genuine = df_genuine.copy()
    df_genuine["cc_num"] = df_genuine["cc_num"].astype(str)
    df_genuine["merchant"] = df_genuine["merchant"].astype(str)
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


def compute_data_profile(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         n_top_cat: int = 12, n_bins: int = 40) -> dict:
    """Compact EDA summary saved with the model so the demo can chart the data
    characteristics without re-reading the raw CSVs. All values are small lists.

    NOTE: counts reflect the rows actually loaded for train/eval (fraud is always
    kept, so the training class balance is intentionally inflated vs. the full
    file). Distribution *shapes* are still representative.
    """
    def _counts(df):
        f = int((df["is_fraud"] == 1).sum())
        return {"genuine": int(len(df) - f), "fraud": f}

    prof = {"class": {"train": _counts(train_df), "test": _counts(test_df)}}

    # transaction-amount (log1p) histogram, genuine vs fraud, on the train rows
    amt = np.log1p(pd.to_numeric(train_df["amt"], errors="coerce").fillna(0.0).clip(lower=0)).to_numpy()
    y = train_df["is_fraud"].to_numpy()
    hi = float(np.percentile(amt, 99.5)) if amt.size else 1.0
    lo = float(amt.min()) if amt.size else 0.0
    bins = np.linspace(lo, hi if hi > lo else lo + 1.0, n_bins + 1)
    g, _ = np.histogram(amt[y == 0], bins=bins)
    fr, _ = np.histogram(amt[y == 1], bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    prof["log_amt_hist"] = {"centers": [float(x) for x in centers],
                            "genuine": [int(x) for x in g], "fraud": [int(x) for x in fr]}

    # fraud rate + volume by hour of day (train)
    hour = pd.to_datetime(train_df["trans_date_trans_time"], errors="coerce").dt.hour.fillna(12).astype(int)
    byh = (pd.DataFrame({"hour": hour, "is_fraud": y})
           .groupby("hour")["is_fraud"].agg(["mean", "size"]).reindex(range(24), fill_value=0))
    prof["hour"] = {"rate": [float(x) for x in byh["mean"]],
                    "count": [int(x) for x in byh["size"]]}

    # fraud rate by merchant category (top categories by volume, train)
    cat = train_df["category"].astype(str)
    dfc = pd.DataFrame({"cat": cat, "is_fraud": y})
    top = dfc["cat"].value_counts().head(n_top_cat).index.tolist()
    grp = dfc[dfc["cat"].isin(top)].groupby("cat")["is_fraud"].agg(["mean", "size"]).reindex(top)
    prof["category"] = {"labels": [str(c) for c in top],
                        "rate": [float(x) for x in grp["mean"]],
                        "count": [int(x) for x in grp["size"]]}
    return prof


def _split_train(df: pd.DataFrame, val_fraction: float, seed: int):
    """Genuine-only training set; validation = held-out genuine + all train fraud."""
    genuine = df[df["is_fraud"] == 0]
    fraud = df[df["is_fraud"] == 1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(genuine))
    n_val = int(len(genuine) * val_fraction)
    val_gen = genuine.iloc[perm[:n_val]]
    train_gen = genuine.iloc[perm[n_val:]]
    val_df = pd.concat([val_gen, fraud]).sample(frac=1.0, random_state=seed)
    return train_gen.reset_index(drop=True), val_df.reset_index(drop=True)


def run_training(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    train_path, test_path = resolve_data_paths(cfg)
    print(f"[data] train={train_path.name} test={test_path.name}")
    train_raw = load_raw(train_path, cfg.train_subsample, cfg.keep_all_fraud, cfg.seed)
    test_raw = load_raw(test_path, cfg.test_subsample, cfg.keep_all_fraud, cfg.seed)
    print(f"[data] train rows={len(train_raw)} fraud={int(train_raw.is_fraud.sum())} | "
          f"test rows={len(test_raw)} fraud={int(test_raw.is_fraud.sum())}")

    train_gen, val_df = _split_train(train_raw, cfg.val_fraction, cfg.seed)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(prefer: str = "auto") -> str:
    """Return 'cuda' when available (e.g. the user's GTX 1650), else 'cpu'."""
    try:
        import torch

        if prefer == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"



HETERO_METADATA = (
    ["customer", "merchant", "transaction"],
    [
        ("customer", "makes", "transaction"),
        ("transaction", "rev_makes", "customer"),
        ("merchant", "sells", "transaction"),
        ("transaction", "rev_sells", "merchant"),
    ],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


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
