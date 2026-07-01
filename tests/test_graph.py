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
    assert set(ref["customer"].keys()) == {"c0", "c1", "c2"}
    vec = next(iter(ref["customer"].values()))
    assert len(vec) == feats["customer"].dim
