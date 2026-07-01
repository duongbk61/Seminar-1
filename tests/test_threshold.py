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
