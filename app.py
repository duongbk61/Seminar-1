"""Streamlit demo for the Heterogeneous Graph Auto-Encoder fraud detector.

Run (after `python main.py` has produced outputs/artifacts.pt):

    streamlit run app.py

The app scores a single transaction in real time - no retraining. You can load a
random genuine / fraud example, pick a known customer & merchant, or invent brand
new ones (cold-start), tweak the amount / hour / category, and watch the model's
reconstruction error cross the fraud threshold.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import Config
from src.scorer import FraudScorer

ARTIFACT = Config().output_dir / "artifacts.pt"

st.set_page_config(page_title="Hetero-Graph Fraud Detector", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_scorer(path_str: str, mtime: float) -> FraudScorer:
    # mtime is part of the cache key so a re-trained model is picked up
    return FraudScorer.load(path_str, device="cpu")


def random_new_customer() -> tuple[str, dict]:
    cc = str(random.randint(4_000_000_000_000_000, 4_999_999_999_999_999))
    attrs = {
        "lat": round(random.uniform(25, 48), 4), "long": round(random.uniform(-122, -71), 4),
        "city_pop": random.randint(1_000, 2_000_000),
        "gender": random.choice(["M", "F"]),
        "dob": f"{random.randint(1955, 2000)}-0{random.randint(1,9)}-1{random.randint(0,9)}",
        "first": "NEW", "last": "CUSTOMER", "city": "Unknown", "state": "NA", "job": "Unknown",
    }
    return cc, attrs


def random_new_merchant(categories: list[str]) -> tuple[str, dict]:
    mid = f"new_Merchant{random.randint(1000, 9999)}"
    attrs = {"category": random.choice(categories) if categories else "shopping_net",
             "merch_lat": round(random.uniform(25, 48), 4),
             "merch_long": round(random.uniform(-122, -71), 4)}
    return mid, attrs


def main() -> None:
    st.title("🛡️ Heterogeneous Graph Auto-Encoder — Credit Card Fraud Detection")
    st.caption("Demo of Majumder et al., *Heterogeneous Graph Auto-Encoder for "
               "Credit Card Fraud Detection* (IJCA 32(2), 2025). The model learns "
               "to reconstruct **genuine** transactions; a high reconstruction "
               "error ⇒ fraud.")

    if not ARTIFACT.exists():
        st.error("No trained model found at `outputs/artifacts.pt`.\n\n"
                 "Train it first:\n\n```\npython main.py\n```")
        st.stop()

    scorer = load_scorer(str(ARTIFACT), ARTIFACT.stat().st_mtime)
    aux = scorer.demo_aux
    cust_attrs = aux.get("customer_attrs", {})
    merch_attrs = aux.get("merchant_attrs", {})
    categories = aux.get("categories", []) or sorted(scorer.feats["merchant"].cat_vocab.get("category", []))
    known_customers = list(cust_attrs.keys())
    known_merchants = list(merch_attrs.keys())

    # ---------------- sidebar: model card ---------------- #
    with st.sidebar:
        st.header("Model performance")
        m = scorer.metrics
        if m:
            st.caption("Full test @ threshold = μ+2σ (paper Eq. 9)")
            st.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
            st.metric("AUC-PR", f"{m.get('auc_pr', float('nan')):.3f}")
            st.metric("F1", f"{m.get('f1', float('nan')):.3f}")
            st.metric("Precision / Recall",
                      f"{m.get('precision', 0):.2f} / {m.get('recall', 0):.2f}")
        if scorer.metrics_f1:
            st.caption(f"(F1-sweep F1 = {scorer.metrics_f1.get('f1', float('nan')):.3f})")
        st.caption(f"Decision threshold = `{scorer.threshold:.6f}`")
        st.caption(f"Known customers: {len(known_customers)} · merchants: {len(known_merchants)}")

    _init_state(scorer, categories)

    _data_characteristics(scorer)
    _training_progress(scorer)

    # ---------------- quick-load example buttons ---------------- #
    st.subheader("1 · Pick a transaction")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("🎲 Random genuine", use_container_width=True):
        _load_example(scorer, "genuine", known_customers, known_merchants)
    if c2.button("🚨 Random fraud", use_container_width=True):
        _load_example(scorer, "fraud", known_customers, known_merchants)
    if c3.button("🆕 New customer", use_container_width=True):
        cc, attrs = random_new_customer()
        st.session_state.update(cc_num=cc, new_cust=attrs, cust_choice="➕ New customer")
        st.rerun()
    if c4.button("🆕 New merchant", use_container_width=True):
        mid, attrs = random_new_merchant(categories)
        st.session_state.update(merchant=mid, new_merch=attrs, merch_choice="➕ New merchant")
        st.rerun()

    # ---------------- entity + transaction panels ---------------- #
    colc, colm, colt = st.columns(3)
    cust = _customer_panel(colc, known_customers, cust_attrs)
    merch = _merchant_panel(colm, known_merchants, merch_attrs, categories)
    txn = _transaction_panel(colt, categories, merch)

    record = _assemble_record(cust, merch, txn)

    st.subheader("2 · Score it")
    if st.button("🔎 Check transaction", type="primary", use_container_width=True):
        _show_result(scorer, record)


# --------------------------------------------------------------------------- #
def _data_characteristics(scorer):
    """Chart the train/test data characteristics saved with the model (EDA)."""
    prof = getattr(scorer, "data_profile", {}) or {}
    if not prof:
        return  # older artifact without a data profile — nothing to show

    with st.expander("📊 Data characteristics (train / test)", expanded=False):
        cls = prof.get("class", {})
        tr, te = cls.get("train", {}), cls.get("test", {})

        def _rate(d):
            n = d.get("genuine", 0) + d.get("fraud", 0)
            return (100.0 * d.get("fraud", 0) / n) if n else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Train rows", f"{tr.get('genuine', 0) + tr.get('fraud', 0):,}",
                  help="Rows loaded for training (all fraud kept)")
        c2.metric("Test rows", f"{te.get('genuine', 0) + te.get('fraud', 0):,}")
        c3.metric("Fraud rate (train / test)", f"{_rate(tr):.2f}% / {_rate(te):.2f}%")

        st.caption("Class balance (genuine vs fraud)")
        st.bar_chart(pd.DataFrame(
            {"genuine": [tr.get("genuine", 0), te.get("genuine", 0)],
             "fraud": [tr.get("fraud", 0), te.get("fraud", 0)]},
            index=pd.Index(["train", "test"], name="split"),
        ))

        left, right = st.columns(2)
        h = prof.get("log_amt_hist", {})
        if h.get("centers"):
            with left:
                st.caption("Amount distribution — log(1+amt), genuine vs fraud")
                st.bar_chart(pd.DataFrame(
                    {"genuine": h["genuine"], "fraud": h["fraud"]},
                    index=pd.Index([round(c, 2) for c in h["centers"]], name="log_amt"),
                ))
        hr = prof.get("hour", {})
        if hr.get("rate"):
            with right:
                st.caption("Fraud rate by hour of day")
                st.bar_chart(pd.DataFrame(
                    {"fraud rate": hr["rate"]},
                    index=pd.Index(list(range(24)), name="hour"),
                ))

        cat = prof.get("category", {})
        if cat.get("labels"):
            st.caption("Fraud rate by merchant category (top categories by volume)")
            st.bar_chart(pd.DataFrame(
                {"fraud rate": cat["rate"]},
                index=pd.Index(cat["labels"], name="category"),
            ))
        st.caption("Counts reflect the rows loaded for train/eval (fraud always kept); "
                   "distribution shapes are representative of the full data.")


def _training_progress(scorer):
    """Visualise the per-epoch training curves saved with the model."""
    hist = getattr(scorer, "history", {}) or {}
    train_loss = hist.get("train_loss") or []
    val_loss = hist.get("val_loss") or []
    val_auc = hist.get("val_auc_pr") or []
    if not train_loss:
        return  # older artifact without history — nothing to show

    with st.expander(f"📈 Training progress ({len(train_loss)} epochs)", expanded=False):
        epochs = list(range(1, len(train_loss) + 1))
        loss_df = pd.DataFrame(
            {"Train loss": train_loss, "Val recon error (genuine)": val_loss},
            index=pd.Index(epochs, name="Epoch"),
        )
        auc_df = pd.DataFrame(
            {"Val AUC-PR": val_auc}, index=pd.Index(epochs, name="Epoch")
        )
        lc, rc = st.columns(2)
        with lc:
            st.caption("Loss per epoch")
            st.line_chart(loss_df)
        with rc:
            st.caption("Validation AUC-PR per epoch")
            st.line_chart(auc_df)
        best_ep = int(max(range(len(val_auc)), key=lambda i: val_auc[i]) + 1) if val_auc else len(epochs)
        st.caption(
            f"Best val AUC-PR = {max(val_auc):.4f} at epoch {best_ep} · "
            f"final train loss = {train_loss[-1]:.4f}"
            if val_auc else f"final train loss = {train_loss[-1]:.4f}"
        )
        _epoch_index = pd.Index(epochs, name="Epoch")

        # threshold vs. the genuine/fraud error bands it separates (paper Eq. 9)
        thr = hist.get("thr_mu2sigma") or []
        gen_mean = hist.get("val_err_genuine_mean") or []
        fraud_mean = hist.get("val_err_fraud_mean") or []
        if thr:
            st.caption("Decision threshold (μ+2σ) vs. genuine/fraud reconstruction error")
            st.line_chart(pd.DataFrame(
                {"Threshold μ+2σ": thr, "Genuine mean error": gen_mean,
                 "Fraud mean error": fraud_mean},
                index=_epoch_index,
            ))

        # validation detection metrics at each epoch's μ+2σ threshold
        val_p = hist.get("val_precision") or []
        val_r = hist.get("val_recall") or []
        val_f1 = hist.get("val_f1") or []
        # per-type reconstruction loss
        type_loss = {k[len("loss_"):]: hist[k] for k in hist if k.startswith("loss_")}
        mc, tc = st.columns(2)
        if val_f1:
            with mc:
                st.caption("Validation P / R / F1 @ μ+2σ per epoch")
                st.line_chart(pd.DataFrame(
                    {"Precision": val_p, "Recall": val_r, "F1": val_f1},
                    index=_epoch_index,
                ))
        if type_loss:
            with tc:
                st.caption("Reconstruction loss per node type")
                st.line_chart(pd.DataFrame(type_loss, index=_epoch_index))


def _init_state(scorer, categories):
    ss = st.session_state
    ss.setdefault("cust_choice", "➕ New customer")
    ss.setdefault("merch_choice", "➕ New merchant")
    if "new_cust" not in ss:
        cc, attrs = random_new_customer(); ss.cc_num = cc; ss.new_cust = attrs
    if "new_merch" not in ss:
        mid, attrs = random_new_merchant(categories); ss.merchant = mid; ss.new_merch = attrs
    ss.setdefault("amt", 75.0)
    ss.setdefault("hour", 14)
    ss.setdefault("txn_date", pd.Timestamp("2020-06-15").date())
    ss.setdefault("category", categories[0] if categories else "shopping_net")


def _load_example(scorer, kind, known_customers, known_merchants):
    examples = scorer.demo_aux.get("examples", {}).get(kind, [])
    if not examples:
        st.warning(f"No {kind} examples stored."); return
    ex = random.choice(examples)
    ss = st.session_state
    ts = pd.to_datetime(ex.get("trans_date_trans_time"), errors="coerce")
    ss.amt = float(ex.get("amt", 75.0))
    ss.category = str(ex.get("category", ss.get("category")))
    ss.hour = int(ts.hour) if pd.notna(ts) else 14
    ss.txn_date = (ts.date() if pd.notna(ts) else pd.Timestamp("2020-06-15").date())

    cc = str(ex.get("cc_num", ""))
    if cc in known_customers:
        ss.cust_choice = cc
    else:
        ss.cust_choice = "➕ New customer"
        ss.cc_num = cc or st.session_state.get("cc_num")
        ss.new_cust = {k: ex.get(k) for k in ("lat", "long", "city_pop", "gender", "dob",
                                              "first", "last", "city", "state", "job")}
    mid = str(ex.get("merchant", ""))
    if mid in known_merchants:
        ss.merch_choice = mid
    else:
        ss.merch_choice = "➕ New merchant"
        ss.merchant = mid or st.session_state.get("merchant")
        ss.new_merch = {"category": ex.get("category"), "merch_lat": ex.get("merch_lat"),
                        "merch_long": ex.get("merch_long")}
    st.rerun()


def _customer_panel(col, known_customers, cust_attrs):
    with col:
        st.markdown("**👤 Customer**")
        options = ["➕ New customer"] + known_customers
        choice = st.selectbox("Cardholder (cc_num)", options, key="cust_choice")
        if choice == "➕ New customer":
            attrs = st.session_state.get("new_cust", {})
            cc = st.session_state.get("cc_num", "0")
            st.info("🆕 New cardholder (no history → cold-start)")
        else:
            attrs = cust_attrs.get(choice, {})
            cc = choice
            st.success("✓ Known cardholder (has genuine profile)")
        st.caption(f"home=({attrs.get('lat')}, {attrs.get('long')}) · "
                   f"gender={attrs.get('gender')} · city_pop={attrs.get('city_pop')}")
        return {"cc_num": cc, **attrs}


def _merchant_panel(col, known_merchants, merch_attrs, categories):
    with col:
        st.markdown("**🏪 Merchant**")
        options = ["➕ New merchant"] + known_merchants
        choice = st.selectbox("Merchant", options, key="merch_choice")
        if choice == "➕ New merchant":
            attrs = st.session_state.get("new_merch", {})
            mid = st.session_state.get("merchant", "new")
            st.info("🆕 New merchant (no history → cold-start)")
        else:
            attrs = merch_attrs.get(choice, {})
            mid = choice
            st.success("✓ Known merchant")
        st.caption(f"category={attrs.get('category')} · "
                   f"loc=({attrs.get('merch_lat')}, {attrs.get('merch_long')})")
        return {"merchant": mid, **attrs}


def _transaction_panel(col, categories, merch):
    with col:
        st.markdown("**💳 Transaction**")
        amt = st.number_input("Amount ($)", min_value=0.0, value=float(st.session_state.amt),
                              step=10.0, key="amt")
        cat_default = merch.get("category") or st.session_state.category
        cat_opts = categories if categories else [cat_default]
        if cat_default not in cat_opts:
            cat_opts = [cat_default] + cat_opts
        category = st.selectbox("Category", cat_opts,
                                index=cat_opts.index(cat_default), key="category")
        hour = st.slider("Hour of day", 0, 23, int(st.session_state.hour), key="hour")
        date = st.date_input("Date", value=st.session_state.txn_date, key="txn_date")
        return {"amt": amt, "category": category, "hour": hour, "date": date}


def _assemble_record(cust, merch, txn) -> dict:
    ts = f"{txn['date']} {int(txn['hour']):02d}:00:00"
    return {
        "cc_num": cust.get("cc_num", "0"),
        "merchant": merch.get("merchant", "new"),
        "category": txn["category"],
        "amt": txn["amt"],
        "gender": cust.get("gender", "M"),
        "city_pop": cust.get("city_pop", 50000),
        "lat": cust.get("lat", 38.0),
        "long": cust.get("long", -95.0),
        "merch_lat": merch.get("merch_lat", 38.0),
        "merch_long": merch.get("merch_long", -95.0),
        "dob": cust.get("dob", "1980-01-01"),
        "trans_date_trans_time": ts,
    }


def _show_result(scorer, record):
    res = scorer.score_transaction(record)
    verdict = res["verdict"]
    err, thr, ratio = res["reconstruction_error"], res["threshold"], res["ratio"]

    if res["is_fraud"]:
        st.error(f"## 🚨 {verdict}")
    else:
        st.success(f"## ✅ {verdict}")

    a, b, c = st.columns(3)
    a.metric("Reconstruction error", f"{err:.6f}")
    b.metric("Threshold", f"{thr:.6f}")
    c.metric("Error / Threshold", f"{ratio:.2f}×",
             delta="above" if ratio >= 1 else "below",
             delta_color="inverse")
    st.progress(min(1.0, ratio / 2.0),
                text=f"Anomaly level {min(ratio,2.0):.2f} / 2.0 (1.0 = threshold)")

    tags = []
    tags.append("known customer" if res["customer_known"] else "🆕 new customer (cold-start)")
    tags.append("known merchant" if res["merchant_known"] else "🆕 new merchant (cold-start)")
    st.caption(" · ".join(tags))

    st.markdown("**Top contributing features (largest reconstruction error):**")
    df = pd.DataFrame(res["top_features"]).rename(
        columns={"feature": "Feature", "error": "Recon error", "value": "Scaled value"})
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.set_index("Feature")["Recon error"])

    with st.expander("Raw record sent to the model"):
        st.json(record)


if __name__ == "__main__":
    main()
