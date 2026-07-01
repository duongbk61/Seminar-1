# Dataset

This project uses the Kaggle dataset **Credit Card Transactions Fraud Detection
Dataset** by *kartik2112* (the "Sparkov" simulator), the same data used in the
paper.

- Kaggle page: https://www.kaggle.com/datasets/kartik2112/fraud-detection

## What to download

Download and unzip, then place these two files **directly in this `data/` folder**:

```
data/
  fraudTrain.csv     (~1.85M rows, ~9.6k fraud)
  fraudTest.csv      (~0.55M rows, ~2.1k fraud)
```

That's it — the code auto-detects them (it also checks an `archive/` folder).
Choose how many rows to draw from these real files at run time with
`python main.py --train-size N --test-size N` (all fraud rows are always kept);
omit `--test-size` for the full test set, or pass `--full` to use everything. The
defaults live in `src/config.py` (`train_subsample` / `test_subsample`).

## Don't have the files yet?

The dataset is **required** — there is no synthetic fallback. Training aborts with
a clear error until `fraudTrain.csv` and `fraudTest.csv` are present in `data/`
(or `archive/`). Download them from the Kaggle page linked above.

## Expected columns

```
"" (row index), trans_date_trans_time, cc_num, merchant, category, amt, first,
last, gender, street, city, state, zip, lat, long, city_pop, job, dob,
trans_num, unix_time, merch_lat, merch_long, is_fraud
```
