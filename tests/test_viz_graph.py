import json

import numpy as np
import pandas as pd

from src.data_prep import build_viz_graph
from src.graph_viz import (graph_3d_html, merge_new_transaction,
                           merge_new_transactions, _cap_transactions,
                           NEW_TXN_ID, NEW_TXN_PREFIX)


def _df(n, fraud_tail=0):
    y = [0] * (n - fraud_tail) + [1] * fraud_tail
    return pd.DataFrame({
        "trans_date_trans_time": ["2020-06-15 14:00:00"] * n,
        "cc_num": [f"c{i % 4}" for i in range(n)],
        "merchant": [f"m{i % 3}" for i in range(n)],
        "category": (["grocery_pos", "shopping_net", "gas_transport"] * n)[:n],
        "amt": np.linspace(5, 800, n), "gender": (["M", "F"] * n)[:n],
        "city_pop": np.linspace(1000, 900000, n),
        "lat": np.linspace(30, 45, n), "long": np.linspace(-120, -75, n),
        "merch_lat": np.linspace(31, 46, n), "merch_long": np.linspace(-119, -74, n),
        "dob": ["1980-01-01"] * n, "city": ["Townsville"] * n, "state": ["NA"] * n,
        "job": ["tester"] * n, "is_fraud": y,
        "trans_num": [f"t{i}" for i in range(n)],
    })


def test_build_viz_graph_structure_and_cap():
    df = _df(50, fraud_tail=10)
    g = build_viz_graph(df, max_nodes=20, seed=0)
    node_ids = {n["id"] for n in g["nodes"]}

    txns = [n for n in g["nodes"] if n["type"] == "transaction"]
    assert len(txns) == 20                       # capped at max_nodes transactions
    assert len(node_ids) == len(g["nodes"])      # ids unique (customers/merchants deduped)

    types = {n["type"] for n in g["nodes"]}
    assert types == {"customer", "merchant", "transaction"}
    for t in txns:
        assert t["is_fraud"] in (0, 1)
        # coordinates for the map view
        assert set(t["geo"]) == {"home_lat", "home_lon", "merch_lat", "merch_lon"}

    # every link endpoint references a real node
    for l in g["links"]:
        assert l["source"] in node_ids and l["target"] in node_ids
        assert l["relation"] in ("makes", "sells")
    # two edges (cust->txn, merch->txn) per sampled transaction
    assert len(g["links"]) == 2 * len(txns)


def test_build_viz_graph_oversamples_fraud():
    # 200 rows, 40 fraud available; ask for 100 txns at 30% fraud
    df = _df(200, fraud_tail=40)
    g = build_viz_graph(df, max_nodes=100, seed=7, fraud_frac=0.3)
    txns = [n for n in g["nodes"] if n["type"] == "transaction"]
    n_fraud = sum(1 for t in txns if t["is_fraud"] == 1)
    assert len(txns) == 100
    assert n_fraud == 30                       # exactly 30% over-sampled


def test_build_viz_graph_is_json_serializable():
    df = _df(30, fraud_tail=5)
    g = build_viz_graph(df, max_nodes=15, seed=1)
    json.dumps(g)   # must not raise (no numpy scalars leaking through)


def test_build_viz_graph_empty():
    g = build_viz_graph(_df(0), max_nodes=10)
    assert g == {"nodes": [], "links": []}


def test_graph_3d_html_embeds_data_and_cdn():
    df = _df(20, fraud_tail=4)
    g = build_viz_graph(df, max_nodes=10, seed=2)
    html = graph_3d_html(g, height=500)
    assert "3d-force-graph" in html          # CDN script present
    assert "ForceGraph3D" in html            # renderer invoked
    assert "txn:" in html                     # graph data injected
    assert "500px" in html                    # height honored


def test_graph_3d_html_empty_graph_notice():
    html = graph_3d_html({"nodes": [], "links": []})
    assert "No graph data" in html


def test_cap_transactions_limits_and_stays_consistent():
    df = _df(60, fraud_tail=10)
    g = build_viz_graph(df, max_nodes=60, seed=3)
    capped = _cap_transactions(g, max_txns=100)          # cap above size -> unchanged count
    assert sum(1 for n in capped["nodes"] if n["type"] == "transaction") == 60

    capped = _cap_transactions(g, max_txns=10)
    node_ids = {n["id"] for n in capped["nodes"]}
    assert sum(1 for n in capped["nodes"] if n["type"] == "transaction") == 10
    for l in capped["links"]:                            # links reference kept nodes only
        assert l["source"] in node_ids and l["target"] in node_ids


def test_cap_transactions_always_keeps_new_node():
    df = _df(40, fraud_tail=8)
    g = build_viz_graph(df, max_nodes=40, seed=8)   # 40 transactions, at the cap
    record = {"cc_num": "zzz_new", "merchant": "zzz_new", "amt": 500.0,
              "category": "shopping_net", "trans_date_trans_time": "2020-06-15 02:00:00"}
    merged = merge_new_transaction(g, record, is_fraud=True, score=0.4, threshold=0.2)
    capped = _cap_transactions(merged, max_txns=10)

    new_nodes = [n for n in capped["nodes"] if n["id"] == NEW_TXN_ID]
    assert len(new_nodes) == 1                       # the injected node survives the cap
    txns = [n for n in capped["nodes"] if n["type"] == "transaction"]
    assert len(txns) == 10                           # still bounded (9 old + 1 new)
    node_ids = {n["id"] for n in capped["nodes"]}
    for l in capped["links"]:
        assert l["source"] in node_ids and l["target"] in node_ids


def test_merge_new_transactions_multiple_distinct_colors():
    df = _df(30, fraud_tail=5)
    g = build_viz_graph(df, max_nodes=15, seed=9)
    entries = [
        {"record": {"cc_num": "aaa", "merchant": "shopA", "amt": 10, "category": "x",
                    "trans_date_trans_time": "2020-06-15 01:00:00"},
         "is_fraud": False, "score": 0.1, "threshold": 0.2, "color": "#ff0000", "seq": 1},
        {"record": {"cc_num": "bbb", "merchant": "shopB", "amt": 20, "category": "y",
                    "trans_date_trans_time": "2020-06-15 02:00:00"},
         "is_fraud": True, "score": 0.5, "threshold": 0.2, "color": "#00ff00", "seq": 2},
    ]
    merged = merge_new_transactions(g, entries)

    new_txns = [n for n in merged["nodes"]
                if n.get("is_new") and n["type"] == "transaction"]
    assert len(new_txns) == 2
    assert {n["id"] for n in new_txns} == {f"{NEW_TXN_PREFIX}1", f"{NEW_TXN_PREFIX}2"}
    assert {n["color"] for n in new_txns} == {"#ff0000", "#00ff00"}   # distinct colours
    # the last-scored transaction is flagged for the camera
    latest = [n for n in new_txns if n.get("is_latest")]
    assert len(latest) == 1 and latest[0]["id"] == f"{NEW_TXN_PREFIX}2"
    # two txns x two edges each
    assert sum(1 for l in merged["links"] if l.get("is_new")) == 4
    # original graph untouched
    assert all(not n.get("is_new") for n in g["nodes"])


def test_merge_new_transactions_shares_entities():
    df = _df(20, fraud_tail=4)
    g = build_viz_graph(df, max_nodes=10, seed=10)
    rec = {"cc_num": "same_cc", "merchant": "same_shop", "amt": 5, "category": "z",
           "trans_date_trans_time": "2020-06-15 03:00:00"}
    entries = [
        {"record": rec, "is_fraud": False, "score": 0.1, "threshold": 0.2, "color": "#111", "seq": 1},
        {"record": rec, "is_fraud": True, "score": 0.3, "threshold": 0.2, "color": "#222", "seq": 2},
    ]
    merged = merge_new_transactions(g, entries)
    # both txns share ONE customer node and ONE merchant node
    assert sum(1 for n in merged["nodes"] if n["id"] == "cust:same_cc") == 1
    assert sum(1 for n in merged["nodes"] if n["id"] == "merch:same_shop") == 1


def test_graph_3d_html_caps_transactions():
    df = _df(50, fraud_tail=5)
    g = build_viz_graph(df, max_nodes=50, seed=4)
    html = graph_3d_html(g, max_txns=10)
    # only 10 txn nodes survive -> injected DATA has exactly 10 "transaction" entries
    assert html.count('"type": "transaction"') == 10


def test_merge_new_transaction_adds_flagged_node_and_links():
    df = _df(30, fraud_tail=5)
    g = build_viz_graph(df, max_nodes=15, seed=5)
    n_before = len(g["nodes"])
    record = {"cc_num": "brand_new_cc", "merchant": "brand_new_merch",
              "amt": 999.0, "category": "shopping_net",
              "trans_date_trans_time": "2020-06-15 03:00:00",
              "gender": "M", "city_pop": 1000}
    merged = merge_new_transaction(g, record, is_fraud=True, score=0.5, threshold=0.2)

    new_txn = [n for n in merged["nodes"] if n["id"] == NEW_TXN_ID]
    assert len(new_txn) == 1 and new_txn[0]["is_new"] == 1 and new_txn[0]["is_fraud"] == 1
    # unknown customer + merchant -> two new entity nodes added (3 new nodes total)
    assert len(merged["nodes"]) == n_before + 3
    new_links = [l for l in merged["links"] if l.get("is_new") == 1]
    assert len(new_links) == 2
    # does not mutate the original graph
    assert len(g["nodes"]) == n_before


def test_merge_new_transaction_reuses_known_entities_and_replaces_prior():
    df = _df(20, fraud_tail=4)
    g = build_viz_graph(df, max_nodes=10, seed=6)
    known_cust = next(n["id"] for n in g["nodes"] if n["type"] == "customer")
    cc = known_cust.split("cust:")[1]
    record = {"cc_num": cc, "merchant": "brand_new_merch", "amt": 5.0,
              "category": "gas_transport", "trans_date_trans_time": "2020-06-15 12:00:00"}

    merged = merge_new_transaction(g, record, is_fraud=False, score=0.1, threshold=0.2)
    # known customer reused (no duplicate), only merchant + txn added -> +2 nodes
    assert len(merged["nodes"]) == len(g["nodes"]) + 2

    # scoring again replaces the previous NEW node rather than stacking
    merged2 = merge_new_transaction(merged, record, is_fraud=True, score=0.3, threshold=0.2)
    assert sum(1 for n in merged2["nodes"] if n["id"] == NEW_TXN_ID) == 1
