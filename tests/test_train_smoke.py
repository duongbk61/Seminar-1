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
