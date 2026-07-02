import numpy as np

from src.evaluate import epoch_diagnostics, mu_2sigma_threshold


def test_mu_2sigma_value():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(mu_2sigma_threshold(s) - (s.mean() + 2 * s.std())) < 1e-9


def test_mu_2sigma_covers_about_95_percent():
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 1.0, size=100_000)
    thr = mu_2sigma_threshold(s)
    frac_below = (s < thr).mean()
    assert 0.95 < frac_below < 0.99


def test_epoch_diagnostics_threshold_matches_mu_2sigma():
    # genuine (label 0) errors low, fraud (label 1) errors high
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 20.0, 30.0])
    labels = np.array([0, 0, 0, 0, 0, 1, 1])
    d = epoch_diagnostics(scores, labels)

    genuine = scores[labels == 0]
    assert abs(d["thr_mu"] - genuine.mean()) < 1e-9
    assert abs(d["thr_sigma"] - genuine.std()) < 1e-9
    assert abs(d["thr_mu2sigma"] - mu_2sigma_threshold(genuine)) < 1e-9
    # the two threshold sources must agree, and fraud must sit above genuine
    assert d["val_err_fraud_mean"] > d["val_err_genuine_mean"]


def test_epoch_diagnostics_keys_and_per_type_loss():
    scores = np.array([1.0, 2.0, 3.0, 10.0])
    labels = np.array([0, 0, 0, 1])
    parts = {"customer": 0.5, "merchant": 0.25, "transaction": 0.75}
    d = epoch_diagnostics(scores, labels, parts)

    for k in ("thr_mu", "thr_sigma", "thr_mu2sigma", "val_precision", "val_recall",
              "val_f1", "val_roc_auc", "val_auc_pr", "val_err_genuine_median",
              "val_err_fraud_median"):
        assert k in d, f"missing diagnostic key: {k}"
    assert d["loss_customer"] == 0.5
    assert d["loss_merchant"] == 0.25
    assert d["loss_transaction"] == 0.75
    # perfect separation here -> the fraud row is caught at mu+2sigma
    assert d["val_recall"] == 1.0


def test_epoch_diagnostics_handles_no_fraud():
    scores = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0])
    d = epoch_diagnostics(scores, labels)
    assert np.isnan(d["val_err_fraud_mean"])
    assert np.isfinite(d["thr_mu2sigma"])
