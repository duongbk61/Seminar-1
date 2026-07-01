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
