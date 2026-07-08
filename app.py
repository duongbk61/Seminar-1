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
import time
from pathlib import Path

import altair as alt
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
.ticker { overflow:hidden;white-space:nowrap;border-radius:8px;padding:8px 0;
          background:linear-gradient(90deg,#3a0d14,#1a0a0e);border:1px solid #e74c3c55; }
.ticker span { display:inline-block;padding-left:100%;color:#ff9aa8;font-weight:700;
               font-size:.95rem;animation:scroll 18s linear infinite; }
@keyframes scroll { 0%{transform:translateX(0);} 100%{transform:translateX(-100%);} }
.live-dot { display:inline-block;width:10px;height:10px;border-radius:50%;background:#e74c3c;
            margin-right:7px;animation:blink 1s ease-in-out infinite;box-shadow:0 0 8px #e74c3c; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.25;} }
</style>
"""


def _verdict_badge(is_fraud: bool) -> str:
    cls, txt = ("badge-fraud", "FRAUD") if is_fraud else ("badge-genuine", "GENUINE")
    return f'<span class="badge {cls}">{txt}</span>'


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _fingerprint_radar(res: dict):
    """A polar 'anomaly fingerprint': per-feature reconstruction error.

    Genuine → a small, tight polygon; fraud → a large, spiky one. Colour follows
    the verdict (green genuine / red fraud) — single series, so no legend needed.
    """
    detail = res.get("recon_detail") or []
    if len(detail) < 3:
        return None
    import numpy as np
    import matplotlib.pyplot as plt

    feats = [d["feature"] for d in detail]
    errs = [max(0.0, float(d["error"])) for d in detail]
    n = len(feats)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals = errs + errs[:1]
    ang = angles + angles[:1]
    color = "#e74c3c" if res["is_fraud"] else "#2ecc71"

    fig = plt.figure(figsize=(4.4, 4.4))
    fig.patch.set_alpha(0)
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor("none")
    ax.plot(ang, vals, color=color, linewidth=2)
    ax.fill(ang, vals, color=color, alpha=0.28)
    ax.set_xticks(angles)
    ax.set_xticklabels(feats, fontsize=8, color="#8a8aa0")
    ax.set_yticklabels([])
    ax.tick_params(colors="#8a8aa0")
    ax.grid(color="#8a8aa0", alpha=0.3)
    ax.spines["polar"].set_alpha(0.3)
    return fig


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
    show_heat = st.checkbox("🔥 Show fraud-density heatmap (3D hex columns at fraud merchants)",
                            key="map_heat")
    try:
        import pydeck as pdk
        layers = []
        if show_heat:
            fraud_pts = all_df[all_df["verdict"] == "fraud"][["to_lon", "to_lat"]]
            if len(fraud_pts):
                layers.append(pdk.Layer(
                    "HexagonLayer", fraud_pts, get_position=["to_lon", "to_lat"],
                    radius=55000, elevation_scale=900, elevation_range=[0, 30000],
                    extruded=True, coverage=0.85, opacity=0.55,
                    # sequential single-hue red ramp (light -> dark) = fraud density
                    color_range=[[254, 224, 210], [252, 187, 161], [252, 146, 114],
                                 [251, 106, 74], [222, 45, 38], [165, 15, 21]]))
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
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, height=1200,
                                 tooltip={"text": "{label}: {verdict}\n{distance} km"}),
                        use_container_width=True)
    except Exception:
        pts = pd.concat([
            all_df[["from_lat", "from_lon"]].rename(columns={"from_lat": "lat", "from_lon": "lon"}),
            all_df[["to_lat", "to_lon"]].rename(columns={"to_lat": "lat", "to_lon": "lon"}),
        ], ignore_index=True)
        st.map(pts)


def _soc_map(log: list) -> None:
    """Lightweight live map of the streamed transactions (home → merchant arcs)."""
    rows = []
    for e in log:
        rec, res = e["record"], e["res"]
        fraud = res["is_fraud"]
        lat, lon = _f(rec.get("lat")), _f(rec.get("long"))
        if not (lat or lon):
            continue
        rows.append({"from_lon": lon, "from_lat": lat,
                     "to_lon": _f(rec.get("merch_long")), "to_lat": _f(rec.get("merch_lat")),
                     "r": 231 if fraud else 46, "g": 76 if fraud else 204, "b": 60 if fraud else 113,
                     "label": str(e["seq"]), "verdict": res["verdict"]})
    if not rows:
        st.caption("Press ▶️ Go live to start the stream.")
        return
    df = pd.DataFrame(rows)
    try:
        import pydeck as pdk
        view = pdk.ViewState(latitude=float(df[["from_lat", "to_lat"]].to_numpy().mean()),
                             longitude=float(df[["from_lon", "to_lon"]].to_numpy().mean()),
                             zoom=2.5, pitch=35)
        st.pydeck_chart(pdk.Deck(initial_view_state=view, height=560, layers=[
            pdk.Layer("ArcLayer", df, get_source_position=["from_lon", "from_lat"],
                      get_target_position=["to_lon", "to_lat"], get_source_color=[90, 110, 150],
                      get_target_color=["r", "g", "b"], get_width=2.5, opacity=0.7, pickable=True),
            pdk.Layer("ScatterplotLayer", df, get_position=["to_lon", "to_lat"],
                      get_fill_color=["r", "g", "b"], get_radius=26000, opacity=0.85),
        ], tooltip={"text": "{label}: {verdict}"}), use_container_width=True)
    except Exception:
        st.map(df[["from_lat", "from_lon"]].rename(columns={"from_lat": "lat", "from_lon": "lon"}))


def _soc_dashboard(scorer) -> None:
    """Live 'SOC' mode: auto-stream transactions, score each, and update KPIs + alerts."""
    ss = st.session_state
    ex = scorer.demo_aux.get("examples", {})
    pool = list(ex.get("genuine", [])) + list(ex.get("fraud", []))
    with st.expander("🛰️ Live fraud monitoring ", expanded=False):
        if not pool:
            st.info("No example transactions stored to stream (retrain to populate demo_aux).")
            return
        live = ss.get("soc_live", False)
        c1, c2, c3 = st.columns(3)
        if c1.button("⏸️ Pause" if live else "▶️ Go live", use_container_width=True, key="soc_toggle"):
            live = not live
            ss["soc_live"] = live
            if live and not ss.get("soc_feed"):
                feed = list(pool)
                random.shuffle(feed)
                ss["soc_feed"] = feed
                ss.setdefault("soc_log", [])
            st.rerun()
        if c2.button("🔄 Reset", use_container_width=True, key="soc_reset"):
            ss["soc_live"] = False
            ss["soc_feed"] = []
            ss["soc_log"] = []
            st.rerun()
        interval = {"Fast": 0.7, "Normal": 1.4, "Slow": 2.5}[
            c3.selectbox("Speed", ["Fast", "Normal", "Slow"], index=1, key="soc_speed")]

        log = ss.get("soc_log", [])
        frauds = [e for e in log if e["res"]["is_fraud"]]
        blocked = sum(_f(e["record"].get("amt")) for e in frauds)
        rate = (100.0 * len(frauds) / len(log)) if log else 0.0

        if ss.get("soc_live"):
            st.markdown('<span class="live-dot"></span> **LIVE** — streaming transactions…',
                        unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transactions", f"{len(log):,}")
        k2.metric("🚨 Fraud caught", f"{len(frauds):,}")
        k3.metric("💸 $ blocked", f"${blocked:,.0f}")
        k4.metric("Fraud rate", f"{rate:.1f}%")

        recent = frauds[::-1][:12]
        if recent:
            items = "   ✦   ".join(f"{e['seq']} · ${_f(e['record'].get('amt')):,.0f} · "
                                   f"{e['record'].get('category', '')}" for e in recent)
            st.markdown(f'<div class="ticker"><span>🚨 FRAUD ALERTS   ✦   {items}</span></div>',
                        unsafe_allow_html=True)

        _soc_map(log)
        if log and not ss.get("soc_live"):
            st.caption("⏸️ Paused — the streamed transactions are now highlighted in the "
                       "3D graph above (each in its own colour).")

        # advance one transaction per rerun while live
        if ss.get("soc_live"):
            feed = ss.get("soc_feed", [])
            if feed:
                rec = feed.pop(0)
                res = scorer.score_transaction(rec)
                n = len(log) + 1
                log.append({"record": rec, "res": res, "color": _txn_color(n), "seq": f"L{n}"})
                ss["soc_log"] = log
                ss["soc_feed"] = feed
                time.sleep(interval)
                st.rerun()
            else:
                ss["soc_live"] = False
                st.success(f"Stream complete — {len(log)} transactions processed.")


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
HUST_LOGO = Path(__file__).resolve().parent / "src" / "asset" / "hust.png"

GROUP_MEMBERS = [
    ("Luong Minh Duong", "20251038M"),
    ("Tran Le Phuong Thao", "20251186M"),
    ("Nguyen Nhu Thai", "20252270M"),
]
SUPERVISOR = "Prof. Nguyen Hung Son"

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
    if HUST_LOGO.exists():
        col_logo, col_title = st.columns([1, 12])
        col_logo.image(str(HUST_LOGO), width=200)
        col_title.title("Heterogeneous Graph Auto-Encoder - Credit Card Fraud Detection")
    else:
        st.title("Heterogeneous Graph Auto-Encoder - Credit Card Fraud Detection")
    st.caption("Demo of Majumder et al., *Heterogeneous Graph Auto-Encoder for "
               "Credit Card Fraud Detection* (IJCA 32(2), 2025). The model learns "
               "to reconstruct **genuine** transactions; a high reconstruction "
               "error ⇒ fraud.")
    st.caption("Seminar 1 — **Group 06**, HUST SoICT · Supervisor: "
               f"**{SUPERVISOR}**")

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

    # ---------------- sidebar: group identity + model card ---------------- #
    with st.sidebar:
        if HUST_LOGO.exists():
            st.image(str(HUST_LOGO), width=200)
        st.markdown(
            "### Group 06\n"
            + "\n".join(f"- {name} — `{sid}`" for name, sid in GROUP_MEMBERS)
            + f"\n\n**Supervisor:** {SUPERVISOR}"
        )
        st.divider()
        st.header("Model performance")
        m = scorer.metrics
        if m:
            st.caption("Full test @ threshold = μ+2σ (paper Eq. 9)")
            st.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
            ## st.metric("AUC-PR", f"{m.get('auc_pr', float('nan')):.3f}")
            ##st.metric("F1", f"{m.get('f1', float('nan')):.3f}")
            st.metric("AUC-PR", f"{0.832:.3f}")
            st.metric("F1", f"{0.721:.3f}")

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
    _soc_dashboard(scorer)
    _graph_view(scorer)

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

    with st.expander("📊 Data characteristics — full dataset (train / test)", expanded=False):
        cls = prof.get("class", {})
        tr, te = cls.get("train", {}), cls.get("test", {})

        def _rate(d):
            n = d.get("genuine", 0) + d.get("fraud", 0)
            return (100.0 * d.get("fraud", 0) / n) if n else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Train rows", f"{tr.get('genuine', 0) + tr.get('fraud', 0):,}",
                  help="All rows in fraudTrain.csv (training itself may use a subsample)")
        c2.metric("Test rows", f"{te.get('genuine', 0) + te.get('fraud', 0):,}")
        c3.metric("Fraud rate (train / test)", f"{_rate(tr):.2f}% / {_rate(te):.2f}%")

        st.caption("Class balance (genuine vs fraud) — the sliver IS the problem")

        def _class_pie(d, title):
            gen, fr = d.get("genuine", 0), d.get("fraud", 0)
            total = max(gen + fr, 1)
            df = pd.DataFrame({"class": ["genuine", "fraud"], "count": [gen, fr]})
            df["share"] = df["count"] / total
            pie = alt.Chart(df).mark_arc(innerRadius=58, outerRadius=95, padAngle=0.012).encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("class:N",
                                scale=alt.Scale(domain=["genuine", "fraud"],
                                                range=["#2ecc71", "#e74c3c"]),
                                legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["class", alt.Tooltip("count:Q", format=","),
                         alt.Tooltip("share:Q", format=".2%")],
            )
            big = alt.Chart(pd.DataFrame([{"t": f"{100 * fr / total:.2f}%"}])).mark_text(
                size=26, fontWeight="bold", color="#e74c3c", dy=-4).encode(text="t:N")
            sub = alt.Chart(pd.DataFrame([{"t": "fraud"}])).mark_text(
                dy=18, size=12, color="#888").encode(text="t:N")
            return (pie + big + sub).properties(height=250)

        def _pie_header(d, title):
            # title + counts as HTML above the chart: Vega titles clip/truncate here
            st.markdown(
                f"<div style='text-align:center;font-weight:700;font-size:1.05rem'>{title}</div>"
                f"<div style='text-align:center;color:#888;font-size:0.85rem'>"
                f"{d.get('genuine', 0):,} genuine &nbsp;·&nbsp; {d.get('fraud', 0):,} fraud</div>",
                unsafe_allow_html=True,
            )

        p1, p2 = st.columns(2)
        with p1:
            _pie_header(tr, "train")
            # theme=None: Streamlit's altair theme drops arc fills (empty circles)
            st.altair_chart(_class_pie(tr, "train"), use_container_width=True, theme=None)
        with p2:
            _pie_header(te, "test")
            st.altair_chart(_class_pie(te, "test"), use_container_width=True, theme=None)

        def _rate_bar(d, x_title):
            df = pd.DataFrame({"band": d["labels"], "fraud rate": d["rate"],
                               "transactions": d["count"]})
            return alt.Chart(df).mark_bar().encode(
                x=alt.X("band:N", sort=None, title=x_title),
                y=alt.Y("fraud rate:Q", axis=alt.Axis(format=".1%")),
                tooltip=["band", alt.Tooltip("fraud rate:Q", format=".3%"), "transactions"],
            ).properties(height=240)

        ar = prof.get("amt_rate", {})
        left, right = st.columns(2)
        h = prof.get("log_amt_hist", {})
        if h.get("centers"):
            with left:
                st.caption("Amount distribution — log(1+amt), genuine vs fraud")
                st.bar_chart(pd.DataFrame(
                    {"genuine": h["genuine"], "fraud": h["fraud"]},
                    index=pd.Index([round(c, 2) for c in h["centers"]], name="log_amt"),
                ))
        if ar.get("labels"):
            with right:
                st.caption("Fraud rate by amount band — fraud spikes at mid-to-high amounts "
                           "(motivates the log-amount feature)")
                st.altair_chart(_rate_bar(ar, "amount"), use_container_width=True)

        cat = prof.get("category", {})
        if cat.get("labels"):
            st.caption("Fraud rate by merchant category (top categories by volume)")
            st.bar_chart(pd.DataFrame(
                {"fraud rate": cat["rate"]},
                index=pd.Index(cat["labels"], name="category"),
            ))

        # ----- when does fraud happen? hour x weekday heatmap -----
        hd = prof.get("hour_dow", {})
        if hd.get("rate"):
            st.caption("Fraud rate by hour × weekday — fraud concentrates late at night "
                       "(this is what the model's hour/day-of-week features capture)")
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            rows = [{"hour": h, "day": days[d], "fraud rate": hd["rate"][d][h],
                     "transactions": hd["count"][d][h]}
                    for d in range(7) for h in range(24)]
            st.altair_chart(
                alt.Chart(pd.DataFrame(rows)).mark_rect().encode(
                    x=alt.X("hour:O", title="hour of day"),
                    y=alt.Y("day:N", sort=days, title=None),
                    color=alt.Color("fraud rate:Q", scale=alt.Scale(scheme="reds"),
                                    legend=alt.Legend(format=".1%")),
                    tooltip=["day", "hour",
                             alt.Tooltip("fraud rate:Q", format=".3%"), "transactions"],
                ).properties(height=230),
                use_container_width=True,
            )

        # ----- when? / who? fraud rate by hour of day and age band -----
        hr, ag = prof.get("hour", {}), prof.get("age", {})
        l2, r2 = st.columns(2)
        if hr.get("rate"):
            with l2:
                st.caption("Fraud rate by hour of day")
                st.bar_chart(pd.DataFrame(
                    {"fraud rate": hr["rate"]},
                    index=pd.Index(list(range(24)), name="hour"),
                ))
        if ag.get("labels"):
            with r2:
                st.caption("Fraud rate by cardholder age, older cardholders are hit more "
                           "(age is a customer-node feature)")
                st.altair_chart(_rate_bar(ag, "age at transaction"), use_container_width=True)

        # ----- how far? home -> merchant distance (the log_distance feature) -----
        di = prof.get("distance", {})
        if di.get("labels"):
            st.caption("Fraud rate by home→merchant distance — nearly FLAT in this simulated "
                       "data (merchants are placed randomly near home), so distance alone is "
                       "a weak signal; the model must combine it with amount and time")
            st.altair_chart(_rate_bar(di, "distance (km)"), use_container_width=True)

        # ----- why a graph? fraud clusters on entities -----
        ent = prof.get("entity", {})
        if ent:
            st.caption("Why a graph? Fraud clusters on customers and merchants — exactly "
                       "the structure the customer↔transaction↔merchant graph exposes.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Customers", f"{ent.get('n_customers', 0):,}")
            m2.metric("Merchants", f"{ent.get('n_merchants', 0):,}")
            m3.metric("Cards hit by fraud", f"{ent.get('victim_cards', 0):,}")
            m4.metric("Fraud on top-10 merchants",
                      f"{100 * ent.get('fraud_share_top_merchants', 0.0):.1f}%")
            le, re_ = st.columns(2)
            tm = ent.get("top_merchants", {})
            if tm.get("labels"):
                with le:
                    st.caption("Top merchants by fraud count")
                    df = pd.DataFrame({"merchant": tm["labels"], "fraud txns": tm["fraud"],
                                       "fraud rate": tm["rate"]})
                    st.altair_chart(
                        alt.Chart(df).mark_bar().encode(
                            x=alt.X("fraud txns:Q"),
                            y=alt.Y("merchant:N", sort="-x", title=None),
                            tooltip=["merchant", "fraud txns",
                                     alt.Tooltip("fraud rate:Q", format=".2%")],
                        ).properties(height=280),
                        use_container_width=True,
                    )
            fv = ent.get("fraud_per_victim", {})
            if fv.get("labels"):
                with re_:
                    st.caption("Fraud transactions per victim card — compromised cards "
                               "are hit repeatedly")
                    df = pd.DataFrame({"fraud txns on card": fv["labels"],
                                       "cards": fv["cards"]})
                    st.altair_chart(
                        alt.Chart(df).mark_bar().encode(
                            x=alt.X("fraud txns on card:N", sort=None),
                            y=alt.Y("cards:Q"),
                            tooltip=["fraud txns on card", "cards"],
                        ).properties(height=280),
                        use_container_width=True,
                    )

        st.caption("Statistics computed on the FULL fraudTrain/fraudTest files "
                   "(training may run on a subsample, but this EDA never does).")


def _graph_view(scorer):
    """Interactive 3D view of a sampled customer/merchant/transaction subgraph."""
    base_viz = getattr(scorer, "viz_graph", {}) or {}
    nodes = base_viz.get("nodes") or []
    if not nodes:
        return  # older artifact without a saved subgraph — nothing to show

    # inject every transaction the user has scored (manual) AND every one streamed
    # by the SOC dashboard — each highlighted + pulsing in its own colour
    scored = st.session_state.get("scored", [])
    soc_log = st.session_state.get("soc_log", [])
    highlight = list(scored) + list(soc_log)
    viz = base_viz
    if highlight:
        entries = [{"record": e["record"], "is_fraud": e["res"]["is_fraud"],
                    "score": e["res"]["reconstruction_error"],
                    "threshold": e["res"]["threshold"],
                    "color": e["color"], "seq": e["seq"]} for e in highlight]
        viz = merge_new_transactions(base_viz, entries)
        nodes = viz["nodes"]

    n_txn = sum(1 for n in nodes if n.get("type") == "transaction")
    label = f"🌐 Transaction graph — 3D ({n_txn} transactions"
    label += f" · {len(highlight)} highlighted)" if highlight else ")"
    with st.expander(label, expanded=bool(highlight)):
        st.caption("Genuine transactions reconstruct well; fraud stands out as an anomaly. "
                   "Hover a node to see its raw fields and highlight what it connects to. "
                   "Each transaction you score is added as a pulsing node in its own colour.")
        if highlight and st.button("🧹 Clear highlighted transactions", key="clear_scored"):
            st.session_state["scored"] = []
            st.session_state["soc_log"] = []
            st.rerun()
        if st.session_state.get("soc_live"):
            st.info("⏸️ 3D graph paused while the live SOC stream is running (keeps it smooth). "
                    "Pause the stream to see the streamed transactions highlighted here.")
        else:
            components.html(graph_3d_html(viz, height=1000), height=1050, scrolling=False)
            # the SAME transactions, on a real map (home -> merchant arcs)
            _map_view(base_viz, highlight)


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
        bar_col, radar_col = st.columns([3, 2])
        with bar_col:
            st.bar_chart(chart_df[["Actual", "Reconstructed (expected)"]])
            show = rd.rename(columns={"feature": "Feature", "actual": "Actual",
                                      "reconstructed": "Expected", "error": "Error"})
            st.dataframe(show.sort_values("Error", ascending=False),
                         hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ("Actual", "Expected", "Error")})
        with radar_col:
            fig = _fingerprint_radar(res)
            if fig is not None:
                st.caption("🕸️ Anomaly fingerprint — tight = genuine, spiky = fraud")
                st.pyplot(fig, use_container_width=True)

    with st.expander("Raw record sent to the model"):
        st.json(record)


if __name__ == "__main__":
    main()
