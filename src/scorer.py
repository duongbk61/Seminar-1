"""Real-time fraud scorer used by the Streamlit demo.

Loads the artifacts saved by training and scores a SINGLE transaction with no
retraining. It builds a 3-node heterogeneous graph (the transaction + its
customer + its merchant), each node featurized by its OWN type featurizer, and
compares the transaction's reconstruction error to the mu+2sigma threshold.

* known customer/merchant -> use the stored genuine feature vector
* new customer/merchant   -> cold-start from the incoming record's own fields
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_prep import TypeFeaturizer, HETERO_METADATA
from .model import HeteroGraphAutoEncoder


class FraudScorer:
    def __init__(self, model, feats, reference_store, threshold, feature_names,
                 metrics, device="cpu", demo_aux=None, is_sample=False,
                 metrics_f1=None, threshold_f1=None):
        self.model = model
        self.feats = feats                      # dict type -> TypeFeaturizer
        self.ref = reference_store
        self.threshold = float(threshold)
        self.threshold_f1 = float(threshold_f1) if threshold_f1 is not None else None
        self.feature_names = feature_names      # dict type -> list
        self.metrics = metrics or {}
        self.metrics_f1 = metrics_f1 or {}
        self.device = device
        self.demo_aux = demo_aux or {}
        self.is_sample = is_sample
        self.txn_cont_names = feats["transaction"].cont_cols
        self.model.eval()

    @classmethod
    def load(cls, artifact_path: str | Path, device: str = "cpu") -> "FraudScorer":
        import torch
        art = torch.load(artifact_path, map_location=device, weights_only=False)
        feats = {k: TypeFeaturizer(**d) for k, d in art["featurizers"].items()}
        model = HeteroGraphAutoEncoder(**art["model_kwargs"])
        model.load_state_dict(art["model_state"])
        model.to(device).eval()
        return cls(model, feats, art["reference_store"], art["threshold_mu2sigma"],
                   art["feature_names"], art.get("metrics", {}), device,
                   demo_aux=art.get("demo_aux", {}), is_sample=art.get("is_sample_data", False),
                   metrics_f1=art.get("metrics_f1", {}), threshold_f1=art.get("threshold_f1"))

    def known_customers(self, limit=None):
        ids = list(self.ref.get("customer", {}).keys())
        return ids[:limit] if limit else ids

    def known_merchants(self, limit=None):
        ids = list(self.ref.get("merchant", {}).keys())
        return ids[:limit] if limit else ids

    def _profile(self, kind, key, fallback):
        table = self.ref.get(kind, {})
        if key in table:
            return np.asarray(table[key], dtype=np.float32), True
        return fallback.astype(np.float32), False

    def _build_single_graph(self, x_txn, cust_vec, merch_vec):
        import torch
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["transaction"].x = torch.tensor(x_txn[None, :], dtype=torch.float32)
        data["customer"].x = torch.tensor(cust_vec[None, :], dtype=torch.float32)
        data["merchant"].x = torch.tensor(merch_vec[None, :], dtype=torch.float32)
        e = torch.tensor([[0], [0]], dtype=torch.long)
        data["customer", "makes", "transaction"].edge_index = e.clone()
        data["transaction", "rev_makes", "customer"].edge_index = e.clone()
        data["merchant", "sells", "transaction"].edge_index = e.clone()
        data["transaction", "rev_sells", "merchant"].edge_index = e.clone()
        return data

    def score_transaction(self, record: dict, top_k: int = 5) -> dict:
        x_txn = self.feats["transaction"].transform_one(record)
        cust_cold = self.feats["customer"].transform_one(record)
        merch_cold = self.feats["merchant"].transform_one(record)
        cust_vec, cust_known = self._profile("customer", str(record.get("cc_num", "")), cust_cold)
        merch_vec, merch_known = self._profile("merchant", str(record.get("merchant", "")), merch_cold)

        data = self._build_single_graph(x_txn, cust_vec, merch_vec)
        error = float(self.model.transaction_scores(data, self.device)[0])
        feat_err = self.model.transaction_feature_errors(data, self.device)
        per_feat = np.array([float(feat_err[i][0]) for i in range(len(self.txn_cont_names))])

        is_fraud = error >= self.threshold
        order = np.argsort(per_feat)[::-1][:top_k]
        top_features = [
            {"feature": self.txn_cont_names[i], "error": float(per_feat[i]),
             "value": float(x_txn[i])}
            for i in order
        ]
        return {
            "reconstruction_error": error,
            "threshold": self.threshold,
            "ratio": error / self.threshold if self.threshold > 0 else float("inf"),
            "is_fraud": bool(is_fraud),
            "verdict": "FRAUD" if is_fraud else "NON-FRAUD",
            "customer_known": cust_known,
            "merchant_known": merch_known,
            "top_features": top_features,
        }
