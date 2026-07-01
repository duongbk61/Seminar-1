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
