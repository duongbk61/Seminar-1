"""Real-time fraud scorer used by the Streamlit demo.

Loads the artifacts saved by training and scores a *single* incoming transaction
without any retraining. It builds a tiny 3-node heterogeneous graph
(transaction + its customer + its merchant) and runs the encoder/decoder:

* known customer/merchant  -> use the stored genuine profile (mean tx vector)
* new  customer/merchant   -> cold-start from the transaction itself

Verdict: reconstruction error >= threshold  =>  FRAUD.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_prep import Featurizer
from .model import HeteroGraphAutoEncoder


class FraudScorer:
    def __init__(self, model, featurizer, reference_store, threshold,
                 feature_names, metrics, device="cpu", demo_aux=None, is_sample=False,
                 resid_var=None, recon_names=None, metrics_balanced=None):
        import torch

        self.model = model
        self.feat = featurizer
        self.ref = reference_store
        self.threshold = float(threshold)
        self.feature_names = feature_names
        self.metrics = metrics or {}
        self.metrics_balanced = metrics_balanced or {}
        self.device = device
        self.demo_aux = demo_aux or {}
        self.is_sample = is_sample
        self.resid_var = None if resid_var is None else torch.as_tensor(resid_var, dtype=torch.float32)
        self.recon_idx = model.recon_idx.tolist()
        self.recon_names = recon_names or [feature_names[i] for i in self.recon_idx]
        self.model.eval()

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, artifact_path: str | Path, device: str = "cpu") -> "FraudScorer":
        import torch

        art = torch.load(artifact_path, map_location=device, weights_only=False)
        feat = Featurizer(**art["featurizer"])
        model = HeteroGraphAutoEncoder(**art["model_kwargs"])
        model.load_state_dict(art["model_state"])
        model.to(device).eval()
        return cls(model, feat, art["reference_store"], art["threshold"],
                   art["feature_names"], art.get("metrics", {}), device,
                   demo_aux=art.get("demo_aux", {}),
                   is_sample=art.get("is_sample_data", False),
                   resid_var=art.get("resid_var"),
                   recon_names=art.get("recon_names"),
                   metrics_balanced=art.get("metrics_balanced"))

    # ------------------------------------------------------------------ #
    def known_customers(self, limit: int | None = None) -> list[str]:
        ids = list(self.ref.get("customer", {}).keys())
        return ids[:limit] if limit else ids

    def known_merchants(self, limit: int | None = None) -> list[str]:
        ids = list(self.ref.get("merchant", {}).keys())
        return ids[:limit] if limit else ids

    def _profile(self, kind: str, key: str, fallback: np.ndarray):
        """Stored genuine profile for a known entity, else cold-start fallback."""
        table = self.ref.get(kind, {})
        if key in table:
            return np.asarray(table[key], dtype=np.float32), True
        return fallback.astype(np.float32), False

    def _build_single_graph(self, x_tx, cust_vec, merch_vec):
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["transaction"].x = torch.tensor(x_tx[None, :], dtype=torch.float32)
        data["customer"].x = torch.tensor(cust_vec[None, :], dtype=torch.float32)
        data["merchant"].x = torch.tensor(merch_vec[None, :], dtype=torch.float32)
        e = torch.tensor([[0], [0]], dtype=torch.long)
        data["customer", "makes", "transaction"].edge_index = e.clone()
        data["transaction", "rev_makes", "customer"].edge_index = e.clone()
        data["merchant", "sells", "transaction"].edge_index = e.clone()
        data["transaction", "rev_sells", "merchant"].edge_index = e.clone()
        return data

    # ------------------------------------------------------------------ #
    def score_transaction(self, record: dict, top_k: int = 5) -> dict:
        """Score one raw transaction record (dict with the Sparkov columns)."""
        x_tx = self.feat.transform_one(record)
        cust_key = str(record.get("cc_num", ""))
        merch_key = str(record.get("merchant", ""))
        cust_vec, cust_known = self._profile("customer", cust_key, x_tx)
        merch_vec, merch_known = self._profile("merchant", merch_key, x_tx)

        data = self._build_single_graph(x_tx, cust_vec, merch_vec)
        pfe = self.model.per_feature_error(data, self.device)[0]   # [F] tensor
        error = float(pfe.mean())     # plain reconstruction error (the paper's score)
        per_feat = pfe.numpy()        # raw per-feature error -> ranks the "why"

        is_fraud = error >= self.threshold
        order = np.argsort(per_feat)[::-1][:top_k]
        top_features = [
            {"feature": self.recon_names[i],
             "error": float(per_feat[i]),
             "value": float(x_tx[self.recon_idx[i]])}
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
