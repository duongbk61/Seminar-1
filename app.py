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
import streamlit.components.v1 as components

from src.config import Config
from src.graph_viz import graph_3d_html, merge_new_transactions
from src.scorer import FraudScorer


# Distinct highlight colours for user-scored transactions. Deliberately excludes
# red/green (reserved for fraud/genuine) so a new txn never looks like a verdict.
_NEW_TXN_COLORS = ["#f59e0b", "#22d3ee", "#e879f9", "#fb923c",
                   "#f472b6", "#facc15", "#d946ef", "#a78bfa"]


def _txn_color(seq: int) -> str:
    """A distinct, non-red/green highlight colour for the seq-th scored transaction."""
    return _NEW_TXN_COLORS[(seq - 1) % len(_NEW_TXN_COLORS)]


def _hex_to_rgb(hex_color: str) -> list:
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


_CSS = """
<style>
.verdict { padding:14px 20px;border-radius:12px;font-size:1.35rem;font-weight:800;
           margin:4px 0 12px;letter-spacing:.3px; }
.verdict-fraud { background:linear-gradient(90deg,#b3122a,#e0243f);color:#ffffff;
                 border:1px solid #ff5a72;animation:pulse 1.15s ease-in-out infinite; }
.verdict-genuine { background:linear-gradient(90deg,#0f7a45,#17a35c);color:#ffffff;
                   border:1px solid #4ee397; }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(231,76,60,.45);}
                   50%{box-shadow:0 0 24px 5px rgba(231,76,60,.65);} }
/* translucent card so it stays readable on BOTH light and dark themes */
div[data-testid="stMetric"] { background:rgba(120,140,200,0.12);
                              border:1px solid rgba(120,140,200,0.32);
                              border-radius:10px;padding:12px 16px; }
table.hist { width:100%;border-collapse:collapse;font-size:.9rem; }
table.hist th,table.hist td { padding:7px 10px;border-bottom:1px solid #2a2a3a;text-align:left; }
table.hist th { color:#9aa0b4;font-weight:600; }
.dot { display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:8px;
       vertical-align:middle;box-shadow:0 0 6px 1px currentColor; }
.badge { padding:2px 9px;border-radius:999px;font-size:.76rem;font-weight:800; }
.badge-fraud { background:#e74c3c22;color:#ff8093;border:1px solid #e74c3c66; }
.badge-genuine { background:#2ecc7122;color:#5be69a;border:1px solid #2ecc7166; }
</style>
"""


def _verdict_badge(is_fraud: bool) -> str:
    cls, txt = ("badge-fraud", "FRAUD") if is_fraud else ("badge-genuine", "GENUINE")
    return f'<span class="badge {cls}">{txt}</span>'


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    from math import radians, sin, cos, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(min(1.0, a)))


def _map_view(base_viz: dict, scored: list) -> None:
    """Map the cardholder-home → merchant arc for the SAME transactions as the 3D graph.

    Sampled graph transactions are drawn faint (green=genuine / red=fraud); the
    user's scored transactions are drawn bold in their own graph colour. The
    home↔merchant distance is an actual model feature, so 'fraud reaches far'
    becomes literal.
    """
    def _row(flat, flon, tlat, tlon, rgb, label, verdict):
        return {"from_lat": flat, "from_lon": flon, "to_lat": tlat, "to_lon": tlon,
                "distance": round(_haversine_km(flat, flon, tlat, tlon)),
                "label": label, "verdict": verdict,
                "r": rgb[0], "g": rgb[1], "b": rgb[2]}

    base_rows = []
    for n in (base_viz or {}).get("nodes", []):
        g = n.get("geo")
        if n.get("type") != "transaction" or not g:
            continue
        fraud = n.get("is_fraud") == 1
        base_rows.append(_row(g["home_lat"], g["home_lon"], g["merch_lat"], g["merch_lon"],
                              [231, 76, 60] if fraud else [46, 204, 113],
                              "sampled txn", "fraud" if fraud else "genuine"))

    new_rows = []
    for e in scored:
        rec, res = e["record"], e["res"]
        try:
            new_rows.append(_row(float(rec["lat"]), float(rec["long"]),
                                 float(rec["merch_lat"]), float(rec["merch_long"]),
                                 _hex_to_rgb(e["color"]), f"TXN {e['seq']}", res["verdict"]))
        except (TypeError, ValueError, KeyError):
            continue

    if not base_rows and not new_rows:
        return  # older artifact without geo and nothing scored yet
    all_df = pd.DataFrame(base_rows + new_rows)

    st.markdown("**🗺️ Where did it happen? Cardholder home → merchant**")
    st.caption("Same transactions as the graph above. Faint arcs = sampled "
               "(🟢 genuine / 🔴 fraud); bold arcs = your scored transactions in "
               "their graph colour. Arc length = home↔merchant distance.")
    try:
        import pydeck as pdk
        layers = []
        if base_rows:
            bdf = pd.DataFrame(base_rows)
            layers.append(pdk.Layer(
                "ArcLayer", bdf, get_source_position=["from_lon", "from_lat"],
                get_target_position=["to_lon", "to_lat"],
                get_source_color=[90, 110, 150], get_target_color=["r", "g", "b"],
                get_width=1.2, opacity=0.35, pickable=True, auto_highlight=True))
        if new_rows:
            ndf = pd.DataFrame(new_rows)
            layers.append(pdk.Layer(
                "ArcLayer", ndf, get_source_position=["from_lon", "from_lat"],
                get_target_position=["to_lon", "to_lat"],
                get_source_color=["r", "g", "b"], get_target_color=["r", "g", "b"],
                get_width=5, opacity=0.95, pickable=True, auto_highlight=True))
            layers.append(pdk.Layer(
                "ScatterplotLayer", ndf, get_position=["to_lon", "to_lat"],
                get_fill_color=["r", "g", "b"], get_radius=40000, opacity=0.95))
        view = pdk.ViewState(
            latitude=float(all_df[["from_lat", "to_lat"]].to_numpy().mean()),
            longitude=float(all_df[["from_lon", "to_lon"]].to_numpy().mean()),
            zoom=2.6, pitch=40)
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, height=680,
                                 tooltip={"text": "{label}: {verdict}\n{distance} km"}),
                        use_container_width=True)
    except Exception:
        pts = pd.concat([
            all_df[["from_lat", "from_lon"]].rename(columns={"from_lat": "lat", "from_lon": "lon"}),
            all_df[["to_lat", "to_lon"]].rename(columns={"to_lat": "lat", "to_lon": "lon"}),
        ], ignore_index=True)
        st.map(pts)


def _history_table(scored: list) -> None:
    """A running log of every transaction the user has scored (colour-matched to the graph)."""
    st.markdown(f"**🧾 Your scored transactions ({len(scored)})** — colours match the graph nodes")
    rows = []
    for e in reversed(scored):          # most recent first
        res, rec = e["res"], e["record"]
        rows.append(
            f'<tr><td><span class="dot" style="color:{e["color"]};background:{e["color"]}"></span>'
            f'TXN {e["seq"]}</td>'
            f'<td>{_verdict_badge(res["is_fraud"])}</td>'
            f'<td>{res["reconstruction_error"]:.4f}</td>'
            f'<td>{res["ratio"]:.2f}×</td>'
            f'<td>${float(rec.get("amt", 0)):,.2f}</td>'
            f'<td>{rec.get("category", "")}</td>'
            f'<td>{rec.get("trans_date_trans_time", "")}</td></tr>'
        )
    st.markdown(
        '<table class="hist"><tr><th>Txn</th><th>Verdict</th><th>Recon error</th>'
        '<th>× thr</th><th>Amount</th><th>Category</th><th>When</th></tr>'
        + "".join(rows) + "</table>",
        unsafe_allow_html=True,
    )

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


# Narrated illustrative cases. Home is set to (40.0, -75.0); "far" merchants sit
# across the country so the distance feature is large. Verdicts are whatever the
# model actually returns — these are crafted to *illustrate*, not to hard-code.
_HOME = {"lat": 40.0, "long": -75.0, "city_pop": 85000, "gender": "F", "dob": "1985-04-12"}
_SCENARIOS = [
    {"icon": "🛒", "label": "Everyday grocery run",
     "desc": "Modest amount, early evening, local grocery store — a typical genuine transaction.",
     "amt": 42.30, "hour": 18, "category": "grocery_pos",
     "cust": dict(_HOME), "merch_lat": 40.03, "merch_long": -75.05},
    {"icon": "🌙", "label": "3 a.m. online spree",
     "desc": "Large online purchase in the middle of the night from a distant merchant.",
     "amt": 1180.0, "hour": 3, "category": "shopping_net",
     "cust": dict(_HOME), "merch_lat": 34.05, "merch_long": -118.24},
    {"icon": "🧪", "label": "Card-testing micro-charge",
     "desc": "A tiny odd-hour charge — the pattern fraudsters use to test a stolen card.",
     "amt": 1.23, "hour": 2, "category": "misc_net",
     "cust": dict(_HOME), "merch_lat": 47.61, "merch_long": -122.33},
    {"icon": "✈️", "label": "Out-of-state big ticket",
     "desc": "A high-value in-store purchase far from the cardholder's home city.",
     "amt": 940.0, "hour": 13, "category": "shopping_pos",
     "cust": dict(_HOME), "merch_lat": 29.76, "merch_long": -95.37},
]


def _load_scenario(scorer, sc: dict) -> None:
    """Populate the builder with a narrated scenario, score it, and add it to the graph."""
    ss = st.session_state
    cc = f"scenario_{sc['label'][:6]}"
    mid = f"scenario_{sc['category']}"
    ss.cust_choice = "➕ New customer"; ss.cc_num = cc
    ss.new_cust = {**sc["cust"], "first": "DEMO", "last": "CASE",
                   "city": "Home", "state": "NA", "job": "n/a"}
    ss.merch_choice = "➕ New merchant"; ss.merchant = mid
    ss.new_merch = {"category": sc["category"],
                    "merch_lat": sc["merch_lat"], "merch_long": sc["merch_long"]}
    ss.amt = float(sc["amt"]); ss.category = sc["category"]; ss.hour = int(sc["hour"])
    ss.txn_date = pd.Timestamp("2020-06-15").date()

    record = {
        "cc_num": cc, "merchant": mid, "category": sc["category"], "amt": float(sc["amt"]),
        "gender": sc["cust"]["gender"], "city_pop": sc["cust"]["city_pop"],
        "lat": sc["cust"]["lat"], "long": sc["cust"]["long"],
        "merch_lat": sc["merch_lat"], "merch_long": sc["merch_long"],
        "dob": sc["cust"]["dob"],
        "trans_date_trans_time": f"2020-06-15 {int(sc['hour']):02d}:00:00",
    }
    scored = ss.setdefault("scored", [])
    scored.append({"record": record, "res": scorer.score_transaction(record),
                   "color": _txn_color(len(scored) + 1), "seq": len(scored) + 1})
    st.rerun()


def random_new_merchant(categories: list[str]) -> tuple[str, dict]:
    mid = f"new_Merchant{random.randint(1000, 9999)}"
    attrs = {"category": random.choice(categories) if categories else "shopping_net",
             "merch_lat": round(random.uniform(25, 48), 4),
             "merch_long": round(random.uniform(-122, -71), 4)}
    return mid, attrs


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
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

        # ---- operating point: pick the decision threshold ----
        thr_opts = {}
        if scorer.threshold_f1 is not None:
            thr_opts["F1-balanced (recommended)"] = scorer.threshold_f1
        thr_opts["μ+2σ (paper Eq. 9)"] = scorer.threshold_mu2sigma
        choice = st.radio(
            "Decision threshold", list(thr_opts.keys()), index=0,
            help="μ+2σ keeps ~97.5% of genuine transactions below it — conservative, so "
                 "it misses a lot of fraud on a weak model. F1-balanced is tuned to balance "
                 "catching fraud against false alarms.",
        )
        scorer.threshold = float(thr_opts[choice])   # applies to all scoring below this run
        st.caption(f"Active threshold = `{scorer.threshold:.6f}`")
        st.caption(f"Known customers: {len(known_customers)} · merchants: {len(known_merchants)}")

    _init_state(scorer, categories)

    _data_characteristics(scorer)
    _training_progress(scorer)
    _graph_view(scorer)
    _latent_view(scorer, st.session_state.get("scored", []))

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

    # ---------------- one-click narrated scenarios ---------------- #
    st.caption("…or try a scenario (loads **and** scores it, adding a node to the graph):")
    sc_cols = st.columns(len(_SCENARIOS))
    for col, sc in zip(sc_cols, _SCENARIOS):
        if col.button(f"{sc['icon']} {sc['label']}", use_container_width=True,
                      help=sc["desc"]):
            _load_scenario(scorer, sc)

    # ---------------- entity + transaction panels ---------------- #
    colc, colm, colt = st.columns(3)
    cust = _customer_panel(colc, known_customers, cust_attrs)
    merch = _merchant_panel(colm, known_merchants, merch_attrs, categories)
    txn = _transaction_panel(colt, categories, merch)

    record = _assemble_record(cust, merch, txn)

    st.subheader("2 · Score it")
    if st.button("🔎 Check transaction", type="primary", use_container_width=True):
        # score, append to the running list (each gets its own random colour), and
        # rerun so the 3D graph (rendered above) picks up the new node on the next pass
        scored = st.session_state.setdefault("scored", [])
        scored.append({
            "record": record, "res": scorer.score_transaction(record),
            "color": _txn_color(len(scored) + 1), "seq": len(scored) + 1,
        })
        st.rerun()
    scored = st.session_state.get("scored", [])
    if scored:
        last = scored[-1]
        _show_result(scorer, last["record"], last["res"])
        _history_table(scored)


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


def _graph_view(scorer):
    """Interactive 3D view of a sampled customer/merchant/transaction subgraph."""
    base_viz = getattr(scorer, "viz_graph", {}) or {}
    nodes = base_viz.get("nodes") or []
    if not nodes:
        return  # older artifact without a saved subgraph — nothing to show

    # inject every transaction the user has scored (each highlighted + pulsing in
    # its own colour)
    scored = st.session_state.get("scored", [])
    viz = base_viz
    if scored:
        entries = [{"record": e["record"], "is_fraud": e["res"]["is_fraud"],
                    "score": e["res"]["reconstruction_error"],
                    "threshold": e["res"]["threshold"],
                    "color": e["color"], "seq": e["seq"]} for e in scored]
        viz = merge_new_transactions(base_viz, entries)
        nodes = viz["nodes"]

    n_txn = sum(1 for n in nodes if n.get("type") == "transaction")
    label = f"🌐 Transaction graph — 3D ({n_txn} transactions"
    label += f" · {len(scored)} of yours highlighted)" if scored else ")"
    with st.expander(label, expanded=bool(scored)):
        st.caption("Genuine transactions reconstruct well; fraud stands out as an anomaly. "
                   "Hover a node to see its raw fields and highlight what it connects to. "
                   "Each transaction you score is added as a pulsing node in its own colour.")
        if scored and st.button("🧹 Clear my transactions from the graph", key="clear_scored"):
            st.session_state["scored"] = []
            st.rerun()
        components.html(graph_3d_html(viz, height=1000), height=1050, scrolling=False)
        # the SAME transactions, on a real map (home -> merchant arcs)
        _map_view(base_viz, scored)


def _latent_view(scorer, scored):
    """2D PCA of learned transaction embeddings — genuine cluster vs fraud outliers."""
    ls = getattr(scorer, "latent_scatter", {}) or {}
    pts = ls.get("points") or []
    if not pts:
        return  # older artifact without a latent projection — nothing to show

    import altair as alt
    base_df = pd.DataFrame([{"x": p["x"], "y": p["y"],
                             "class": "fraud" if p["is_fraud"] == 1 else "genuine"}
                            for p in pts])
    with st.expander("🧭 Latent space — what the model learned (2D PCA of embeddings)",
                     expanded=bool(scored)):
        st.caption("Each dot is a transaction's learned embedding projected to 2D. "
                   "Genuine transactions cluster together; fraud tends to sit apart — "
                   "that separation is what the reconstruction error captures. Your scored "
                   "transactions appear as ▲ in their graph colour.")
        base = alt.Chart(base_df).mark_circle(size=55, opacity=0.45).encode(
            x=alt.X("x", title="PC-1"), y=alt.Y("y", title="PC-2"),
            color=alt.Color("class", scale=alt.Scale(domain=["genuine", "fraud"],
                                                      range=["#2ecc71", "#e74c3c"]),
                            legend=alt.Legend(title="sampled")),
            tooltip=["class"])
        layers = [base]
        srows = [{"x": e["res"]["embed_2d"][0], "y": e["res"]["embed_2d"][1],
                  "label": f"TXN {e['seq']}", "color": e["color"]}
                 for e in scored if e["res"].get("embed_2d")]
        if srows:
            sdf = pd.DataFrame(srows)
            layers.append(alt.Chart(sdf).mark_point(
                shape="triangle-up", size=400, filled=True, stroke="white", strokeWidth=1.5
            ).encode(x="x", y="y", color=alt.Color("color:N", scale=None, legend=None),
                     tooltip=["label"]))
        st.altair_chart(alt.layer(*layers).properties(height=480).interactive(),
                        use_container_width=True)


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


def _show_result(scorer, record, res=None):
    res = res if res is not None else scorer.score_transaction(record)
    verdict = res["verdict"]
    err, thr, ratio = res["reconstruction_error"], res["threshold"], res["ratio"]

    if res["is_fraud"]:
        st.markdown(f'<div class="verdict verdict-fraud">🚨 {verdict} — reconstruction error '
                    f'{ratio:.2f}× the threshold</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict verdict-genuine">✅ {verdict} — reconstructs cleanly '
                    f'({ratio:.2f}× the threshold)</div>', unsafe_allow_html=True)

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

    # ---- why? expected vs actual reconstruction (the core of the paper) ----
    detail = res.get("recon_detail")
    if detail:
        st.markdown("**🔍 Why? Expected vs. actual (per feature, model's standardized view)**")
        st.caption("The model reconstructs what a *genuine* transaction should look like. "
                   "Where **actual** diverges from **reconstructed (expected)**, error accrues — "
                   "big gaps drive the anomaly score.")
        rd = pd.DataFrame(detail)
        chart_df = rd.rename(columns={"reconstructed": "Reconstructed (expected)",
                                      "actual": "Actual"}).set_index("feature")
        st.bar_chart(chart_df[["Actual", "Reconstructed (expected)"]])
        show = rd.rename(columns={"feature": "Feature", "actual": "Actual",
                                  "reconstructed": "Expected", "error": "Error"})
        st.dataframe(show.sort_values("Error", ascending=False),
                     hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ("Actual", "Expected", "Error")})

    with st.expander("Raw record sent to the model"):
        st.json(record)


if __name__ == "__main__":
    main()
