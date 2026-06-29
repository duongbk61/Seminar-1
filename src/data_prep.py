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

Heterogeneous graph (paper Section 3): three node types
    customer  (cc_num)
    merchant  (merchant)
    transaction (trans_num)   <- the nodes we reconstruct
with bidirectional edges  customer <-> transaction <-> merchant.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ----- columns the featuriser depends on (must be present in any input record) -----
REQUIRED_COLS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "city_pop", "lat", "long", "merch_lat", "merch_long", "dob",
]

CONT_COLS = ["log_amt", "age", "log_city_pop", "log_distance"]
CYC_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
# The auto-encoder reconstructs ONLY these behavioural features (amount + time).
# Everything else (categories, gender, age, location, day-of-week) is ENCODER
# CONTEXT only: those are either inherently unpredictable by MSE or carry no fraud
# signal, so reconstructing them just adds a constant error floor that dilutes the
# anomaly score. EDA on the real data: amt separates fraud by ~1.6 sigma, hour by
# ~1.0 sigma; distance/city_pop are ~0.01 (useless).
RECON_COLS = ["log_amt", "hour_sin", "hour_cos"]


# --------------------------------------------------------------------------- #
# path resolution + loading
# --------------------------------------------------------------------------- #
def resolve_data_paths(cfg) -> tuple[Path, Path, bool]:
    """Prefer the real Kaggle files (data/ or archive/); fall back to synthetic."""
    project_root = cfg.train_csv.parent.parent
    real_candidates = [
        (cfg.train_csv, cfg.test_csv),
        (project_root / "archive" / "fraudTrain.csv", project_root / "archive" / "fraudTest.csv"),
        (cfg.train_csv.parent / "archive" / "fraudTrain.csv",
         cfg.train_csv.parent / "archive" / "fraudTest.csv"),
    ]
    for train_p, test_p in real_candidates:
        if train_p.exists() and test_p.exists():
            return train_p, test_p, False

    sample_train = cfg.train_csv.parent / "sample" / "fraudTrain.csv"
    sample_test = cfg.test_csv.parent / "sample" / "fraudTest.csv"
    if sample_train.exists() and sample_test.exists():
        warnings.warn(
            "Real fraudTrain.csv & fraudTest.csv not found (looked in data/ and "
            "archive/) - using synthetic sample data in data/sample/."
        )
        return sample_train, sample_test, True
    raise FileNotFoundError(
        "No dataset found. Put fraudTrain.csv & fraudTest.csv in data/ or archive/, "
        "or run `python make_sample_data.py` to create synthetic sample data."
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


def _raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw records into the intermediate numeric/categorical columns."""
    out = pd.DataFrame(index=df.index)
    amt = pd.to_numeric(df["amt"], errors="coerce").fillna(0.0)
    out["log_amt"] = np.log1p(amt.clip(lower=0))

    ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    hour = ts.dt.hour.fillna(12).astype(float)
    dow = ts.dt.dayofweek.fillna(0).astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    dob = pd.to_datetime(df["dob"], errors="coerce")
    age = (ts.dt.year - dob.dt.year)
    out["age"] = age.fillna(age.median() if age.notna().any() else 40).astype(float)

    pop = pd.to_numeric(df["city_pop"], errors="coerce").fillna(0.0)
    out["log_city_pop"] = np.log1p(pop.clip(lower=0))

    dist = _haversine_km(
        pd.to_numeric(df["lat"], errors="coerce").fillna(0.0).values,
        pd.to_numeric(df["long"], errors="coerce").fillna(0.0).values,
        pd.to_numeric(df["merch_lat"], errors="coerce").fillna(0.0).values,
        pd.to_numeric(df["merch_long"], errors="coerce").fillna(0.0).values,
    )
    out["log_distance"] = np.log1p(np.clip(dist, 0, None))

    out["category"] = df["category"].astype(str).values
    out["gender_M"] = (df["gender"].astype(str).str.upper() == "M").astype(float).values
    return out


@dataclass
class Featurizer:
    """Fit on the genuine training rows, then transform any records (incl. one row)."""
    cat_vocab: list[str] = field(default_factory=list)
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "Featurizer":
        raw = _raw_frame(df)
        self.cat_vocab = sorted(raw["category"].unique().tolist())
        cont = raw[CONT_COLS].to_numpy(dtype=np.float64)
        self.mean_ = cont.mean(axis=0)
        std = cont.std(axis=0)
        std[std < 1e-8] = 1.0
        self.scale_ = std
        self.feature_names = (
            list(CONT_COLS) + list(CYC_COLS)
            + [f"cat={c}" for c in self.cat_vocab] + ["gender_M"]
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        raw = _raw_frame(df)
        cont = (raw[CONT_COLS].to_numpy(dtype=np.float64) - self.mean_) / self.scale_
        cyc = raw[CYC_COLS].to_numpy(dtype=np.float64)
        # one-hot category against the fitted vocabulary (unknown -> all zeros)
        idx = {c: i for i, c in enumerate(self.cat_vocab)}
        oh = np.zeros((len(raw), len(self.cat_vocab)), dtype=np.float64)
        for r, c in enumerate(raw["category"].values):
            if c in idx:
                oh[r, idx[c]] = 1.0
        gender = raw[["gender_M"]].to_numpy(dtype=np.float64)
        X = np.hstack([cont, cyc, oh, gender]).astype(np.float32)
        return X

    def transform_one(self, record: dict) -> np.ndarray:
        return self.transform(pd.DataFrame([record]))[0]

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    @property
    def recon_idx(self) -> list[int]:
        """Column indices (into the feature vector) of the reconstruction targets."""
        name_to_i = {n: i for i, n in enumerate(self.feature_names)}
        return [name_to_i[c] for c in RECON_COLS]

    @property
    def recon_names(self) -> list[str]:
        return list(RECON_COLS)


# --------------------------------------------------------------------------- #
# graph construction
# --------------------------------------------------------------------------- #
def _mean_by_group(X: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    F = X.shape[1]
    sums = np.zeros((n_groups, F), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.float64)
    np.add.at(sums, group_idx, X)
    np.add.at(counts, group_idx, 1.0)
    counts[counts == 0] = 1.0
    return (sums / counts[:, None]).astype(np.float32)


def build_hetero_data(df: pd.DataFrame, feat: Featurizer):
    """Return (HeteroData, info) with mean-aggregated customer/merchant features."""
    import torch
    from torch_geometric.data import HeteroData

    X_tx = feat.transform(df)                       # [N, F]
    cust_uni, cust_idx = np.unique(df["cc_num"].values, return_inverse=True)
    merch_uni, merch_idx = np.unique(df["merchant"].values, return_inverse=True)
    tx_idx = np.arange(len(df))

    cust_x = _mean_by_group(X_tx, cust_idx, len(cust_uni))
    merch_x = _mean_by_group(X_tx, merch_idx, len(merch_uni))

    data = HeteroData()
    data["customer"].x = torch.from_numpy(cust_x)
    data["merchant"].x = torch.from_numpy(merch_x)
    data["transaction"].x = torch.from_numpy(X_tx)
    data["transaction"].y = torch.tensor(df["is_fraud"].to_numpy(dtype=np.int64))

    c2t = np.vstack([cust_idx, tx_idx]).astype(np.int64)
    m2t = np.vstack([merch_idx, tx_idx]).astype(np.int64)
    data["customer", "makes", "transaction"].edge_index = torch.from_numpy(c2t)
    data["transaction", "rev_makes", "customer"].edge_index = torch.from_numpy(c2t[[1, 0]])
    data["merchant", "sells", "transaction"].edge_index = torch.from_numpy(m2t)
    data["transaction", "rev_sells", "merchant"].edge_index = torch.from_numpy(m2t[[1, 0]])

    info = {
        "customer_ids": cust_uni,
        "merchant_ids": merch_uni,
        "labels": df["is_fraud"].to_numpy(dtype=np.int64),
        "n_tx": len(df),
    }
    return data, info


def compute_reference_store(df_genuine: pd.DataFrame, feat: Featurizer) -> dict:
    """Profiles of known customers/merchants for the demo (mean genuine tx vector)."""
    X = feat.transform(df_genuine)
    out = {"feature_dim": int(X.shape[1]), "global_mean": X.mean(axis=0).astype(np.float32)}
    for key, col in (("customer", "cc_num"), ("merchant", "merchant")):
        uni, inv = np.unique(df_genuine[col].astype(str).values, return_inverse=True)
        means = _mean_by_group(X, inv, len(uni))
        counts = np.bincount(inv, minlength=len(uni))
        out[key] = {str(k): means[i] for i, k in enumerate(uni)}
        out[key + "_count"] = {str(k): int(counts[i]) for i, k in enumerate(uni)}
    return out


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


HETERO_METADATA = (
    ["customer", "merchant", "transaction"],
    [
        ("customer", "makes", "transaction"),
        ("transaction", "rev_makes", "customer"),
        ("merchant", "sells", "transaction"),
        ("transaction", "rev_sells", "merchant"),
    ],
)
