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


HETERO_METADATA = (
    ["customer", "merchant", "transaction"],
    [
        ("customer", "makes", "transaction"),
        ("transaction", "rev_makes", "customer"),
        ("merchant", "sells", "transaction"),
        ("transaction", "rev_sells", "merchant"),
    ],
)
