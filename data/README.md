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

That's it — the code auto-detects them. By default we subsample
(~120k train / ~40k test, all fraud kept) so training is fast on CPU/GTX 1650.
Change `train_subsample` / `test_subsample` in `src/config.py` (set to `None`
for the full dataset).

## Don't have the files yet?

You can build and run the **entire** pipeline + demo on synthetic data that has
the identical schema:

```bash
python make_sample_data.py          # writes data/sample/fraudTrain.csv & fraudTest.csv
```

When the real `data/fraudTrain.csv` / `data/fraudTest.csv` appear, they take
priority automatically over the synthetic ones.

## Expected columns

```
"" (row index), trans_date_trans_time, cc_num, merchant, category, amt, first,
last, gender, street, city, state, zip, lat, long, city_pop, job, dob,
trans_num, unix_time, merch_lat, merch_long, is_fraud
```
