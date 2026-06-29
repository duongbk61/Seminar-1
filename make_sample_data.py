"""Generate small synthetic credit-card transaction CSVs that mimic the schema of
the Kaggle `kartik2112/fraud-detection` (Sparkov) dataset.

Purpose: let us build and test the whole pipeline (graph construction, training,
scoring, Streamlit demo) BEFORE the real CSVs are downloaded. The synthetic data
contains a *learnable* fraud signal so the model produces sensible metrics:
fraudulent rows tend to have larger amounts, odd hours, larger geo distance
between cardholder and merchant, and favour a few online categories.

Output:
    data/sample/fraudTrain.csv
    data/sample/fraudTest.csv

These have the exact same columns as the real files, so `data_prep.py` reads them
identically. When the real `data/fraudTrain.csv` / `data/fraudTest.csv` exist they
take priority automatically.

Usage:
    python make_sample_data.py --train 20000 --test 6000
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = [
    "grocery_pos", "shopping_net", "misc_net", "shopping_pos", "gas_transport",
    "grocery_net", "entertainment", "food_dining", "health_fitness", "kids_pets",
    "home", "personal_care", "misc_pos", "travel",
]
# Categories where fraud is over-represented in this synthetic generator.
FRAUD_PRONE = {"shopping_net", "misc_net", "grocery_pos"}

# Per-category typical (mean, sigma) for a lognormal amount distribution.
CAT_AMT = {
    "grocery_pos": (3.8, 0.6), "shopping_net": (4.2, 0.8), "misc_net": (3.9, 0.9),
    "shopping_pos": (4.0, 0.7), "gas_transport": (3.5, 0.4), "grocery_net": (3.7, 0.5),
    "entertainment": (3.6, 0.7), "food_dining": (3.3, 0.5), "health_fitness": (3.6, 0.6),
    "kids_pets": (3.5, 0.6), "home": (4.1, 0.8), "personal_care": (3.2, 0.5),
    "misc_pos": (3.6, 0.7), "travel": (4.6, 0.9),
}
JOBS = [
    "Engineer", "Teacher", "Nurse", "Accountant", "Designer", "Chef", "Lawyer",
    "Scientist", "Doctor", "Analyst", "Manager", "Technician", "Artist", "Clerk",
]
STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
GENDERS = ["M", "F"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_entities(rng: np.random.Generator, n_customers: int, n_merchants: int):
    customers = []
    for i in range(n_customers):
        customers.append({
            "cc_num": 4000_0000_0000_0000 + int(rng.integers(0, 9_000_000_000_000)),
            "first": f"Cust{i}", "last": f"L{i % 97}",
            "gender": GENDERS[int(rng.integers(0, 2))],
            "street": f"{int(rng.integers(1, 9999))} Main St",
            "city": f"City{i % 120}", "state": STATES[int(rng.integers(0, len(STATES)))],
            "zip": int(rng.integers(10000, 99999)),
            "lat": float(rng.uniform(25, 48)), "long": float(rng.uniform(-122, -71)),
            "city_pop": int(np.exp(rng.uniform(7, 14))),
            "job": JOBS[int(rng.integers(0, len(JOBS)))],
            "dob": f"{int(rng.integers(1950, 2000))}-{int(rng.integers(1,13)):02d}-{int(rng.integers(1,28)):02d}",
        })
    merchants = []
    for j in range(n_merchants):
        merchants.append({
            "merchant": f"fraud_Merchant{j}",
            "category": CATEGORIES[int(rng.integers(0, len(CATEGORIES)))],
            "mlat": float(rng.uniform(25, 48)), "mlong": float(rng.uniform(-122, -71)),
        })
    return customers, merchants


def _gen_split(rng, customers, merchants, n_rows, start_ts, fraud_rate):
    n_cust, n_merch = len(customers), len(merchants)
    rows = []
    span_seconds = 365 * 24 * 3600
    for _ in range(n_rows):
        c = customers[int(rng.integers(0, n_cust))]
        is_fraud = rng.random() < fraud_rate

        if is_fraud:
            # Fraud favours online categories.
            if rng.random() < 0.7:
                cat = list(FRAUD_PRONE)[int(rng.integers(0, len(FRAUD_PRONE)))]
                m_candidates = [m for m in merchants if m["category"] == cat]
                m = m_candidates[int(rng.integers(0, len(m_candidates)))] if m_candidates \
                    else merchants[int(rng.integers(0, n_merch))]
            else:
                m = merchants[int(rng.integers(0, n_merch))]
            cat = m["category"]
            mu, sigma = CAT_AMT[cat]
            amt = float(np.exp(rng.normal(mu + 1.2, sigma)))   # larger amounts
            hour = int(rng.choice([0, 1, 2, 3, 4, 23], p=[.2, .2, .2, .15, .15, .1]))  # odd hours
            # merchant far from cardholder home
            mlat = c["lat"] + rng.normal(0, 6.0)
            mlong = c["long"] + rng.normal(0, 6.0)
        else:
            m = merchants[int(rng.integers(0, n_merch))]
            cat = m["category"]
            mu, sigma = CAT_AMT[cat]
            amt = float(np.exp(rng.normal(mu, sigma)))
            hour = int(rng.integers(6, 23))                    # daytime
            mlat = m["mlat"] + rng.normal(0, 0.5)
            mlong = m["mlong"] + rng.normal(0, 0.5)

        offset = int(rng.integers(0, span_seconds))
        ts = start_ts + pd.Timedelta(seconds=offset)
        ts = ts.replace(hour=hour, minute=int(rng.integers(0, 60)), second=int(rng.integers(0, 60)))
        trans_num = hashlib.md5(f"{c['cc_num']}{m['merchant']}{ts}{amt}".encode()).hexdigest()
        rows.append({
            "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": c["cc_num"], "merchant": m["merchant"], "category": cat,
            "amt": round(amt, 2), "first": c["first"], "last": c["last"],
            "gender": c["gender"], "street": c["street"], "city": c["city"],
            "state": c["state"], "zip": c["zip"], "lat": c["lat"], "long": c["long"],
            "city_pop": c["city_pop"], "job": c["job"], "dob": c["dob"],
            "trans_num": trans_num, "unix_time": int(ts.timestamp()),
            "merch_lat": round(mlat, 6), "merch_long": round(mlong, 6),
            "is_fraud": int(is_fraud),
        })
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--test", type=int, default=6000)
    ap.add_argument("--customers", type=int, default=400)
    ap.add_argument("--merchants", type=int, default=200)
    ap.add_argument("--fraud-rate", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = _rng(args.seed)
    customers, merchants = _make_entities(rng, args.customers, args.merchants)

    out_dir = Path(__file__).resolve().parent / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _gen_split(rng, customers, merchants, args.train,
                          pd.Timestamp("2019-01-01"), args.fraud_rate)
    test_df = _gen_split(rng, customers, merchants, args.test,
                         pd.Timestamp("2020-01-01"), args.fraud_rate)

    # First (index) column is unnamed in the real dataset.
    train_df.to_csv(out_dir / "fraudTrain.csv", index=True)
    test_df.to_csv(out_dir / "fraudTest.csv", index=True)

    def _summ(name, df):
        n, f = len(df), int(df["is_fraud"].sum())
        print(f"{name}: {n} rows, {f} fraud ({100*f/n:.2f}%)")

    print(f"Wrote synthetic sample data to {out_dir}")
    _summ("train", train_df)
    _summ("test", test_df)


if __name__ == "__main__":
    main()
