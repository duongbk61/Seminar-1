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
