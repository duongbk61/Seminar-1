"""Train the hetero-graph auto-encoder ONCE on genuine transactions, then save
everything the demo needs (weights, featuriser, reference store, threshold).

Run via `python main.py` (which calls `run_training`).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_prep, evaluate
from .config import Config
from .model import ae_loss, build_model
from .utils import pick_device, set_seed

ARTIFACT_PATH_NAME = "artifacts.pt"


def _split_train(df: pd.DataFrame, val_fraction: float, seed: int):
    """Genuine-only training set; validation = held-out genuine + all train fraud."""
    genuine = df[df["is_fraud"] == 0]
    fraud = df[df["is_fraud"] == 1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(genuine))
    n_val = int(len(genuine) * val_fraction)
    val_gen = genuine.iloc[perm[:n_val]]
    train_gen = genuine.iloc[perm[n_val:]]
    val_df = pd.concat([val_gen, fraud]).sample(frac=1.0, random_state=seed)
    return train_gen.reset_index(drop=True), val_df.reset_index(drop=True)


def run_training(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    train_path, test_path, is_sample = data_prep.resolve_data_paths(cfg)
    print(f"[data] train={train_path.name} test={test_path.name} (synthetic={is_sample})")

    train_raw = data_prep.load_raw(train_path, cfg.train_subsample, cfg.keep_all_fraud, cfg.seed)
    test_raw = data_prep.load_raw(test_path, cfg.test_subsample, cfg.keep_all_fraud, cfg.seed)
    print(f"[data] train rows={len(train_raw)} fraud={int(train_raw.is_fraud.sum())} | "
          f"test rows={len(test_raw)} fraud={int(test_raw.is_fraud.sum())}")

    train_gen, val_df = _split_train(train_raw, cfg.val_fraction, cfg.seed)

    # featuriser is fit on genuine training rows only (no leakage)
    feat = data_prep.Featurizer().fit(train_gen)
    print(f"[feat] {feat.dim} features: {feat.feature_names}")

    train_data, _ = data_prep.build_hetero_data(train_gen, feat)
    val_data, val_info = data_prep.build_hetero_data(val_df, feat)
    test_data, test_info = data_prep.build_hetero_data(test_raw, feat)

    metadata = train_data.metadata()
    in_dims = {nt: feat.dim for nt in metadata[0]}
    recon_idx = feat.recon_idx             # reconstruct only amount + time features
    model_kwargs = dict(
        metadata=metadata, in_dims=in_dims, recon_idx=recon_idx,
        hidden_dim=cfg.hidden_dim, heads=cfg.heads, encoder_layers=cfg.encoder_layers,
        latent_dim=cfg.latent_dim, decoder_hidden=cfg.decoder_hidden, dropout=cfg.dropout,
    )
    model = build_model(cfg, metadata, in_dims, recon_idx).to(device)
    opt = __import__("torch").optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    import torch
    train_data = train_data.to(device)
    target = train_data["transaction"].x[:, recon_idx]
    val_labels = val_info["labels"]

    history = {"train_loss": [], "val_loss": [], "val_auc_pr": []}
    best_auc, best_state, patience = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        recon, mu, logvar = model(train_data.x_dict, train_data.edge_index_dict)
        loss, recon_loss, kl = ae_loss(recon, target, mu, logvar, cfg.beta)
        loss.backward()
        opt.step()

        pfe_val = model.per_feature_error(val_data, device).numpy()
        gen = val_labels == 0
        val_scores = pfe_val.mean(axis=1)          # unweighted recon error (paper's score)
        val_gen_loss = float(pfe_val[gen].mean())
        try:
            from sklearn.metrics import average_precision_score
            val_auc = float(average_precision_score(val_labels, val_scores))
        except Exception:
            val_auc = float("nan")

        history["train_loss"].append(float(recon_loss.item()))
        history["val_loss"].append(val_gen_loss)
        history["val_auc_pr"].append(val_auc)

        improved = val_auc > best_auc + 1e-4
        if improved:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"[epoch {epoch:3d}] train_loss={recon_loss.item():.5f} "
                  f"kl={kl.item():.4f} val_recon={val_gen_loss:.5f} val_auc_pr={val_auc:.4f}")
        if patience >= cfg.early_stop_patience:
            print(f"[early stop] no val improvement for {cfg.early_stop_patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[train] done in {time.time()-t0:.1f}s, best val AUC-PR={best_auc:.4f}")

    # genuine per-feature residual variance - kept only to rank features in the
    # demo's "why" explanation (NOT used for the verdict, which is the plain
    # reconstruction error as in the paper).
    resid_var = model.per_feature_error(train_data, device).numpy().mean(axis=0) + 1e-6

    def score(data):
        return model.per_feature_error(data, device).numpy().mean(axis=1)

    # ----- threshold from validation, final metrics on the test graph ----- #
    val_scores = score(val_data)
    threshold, val_f1 = evaluate.search_threshold(val_scores, val_labels)
    test_scores = score(test_data)
    test_labels = test_info["labels"]
    metrics = evaluate.compute_metrics(test_scores, test_labels, threshold)
    print(f"[eval] threshold={threshold:.6f} (val F1={val_f1:.3f})")
    print(f"[eval] FULL TEST (imbalanced {100*test_labels.mean():.2f}% fraud)  "
          f"AUC-PR={metrics['auc_pr']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  "
          f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f}")

    # ----- balanced test view (interpretable F1, comparable to the paper) ----- #
    rng = np.random.default_rng(cfg.seed)
    fidx = np.where(test_labels == 1)[0]
    gidx = rng.choice(np.where(test_labels == 0)[0], size=len(fidx), replace=False)
    bidx = np.concatenate([fidx, gidx])
    bal_scores, bal_labels = test_scores[bidx], test_labels[bidx]
    bal_thr, _ = evaluate.search_threshold(bal_scores, bal_labels)
    metrics_balanced = evaluate.compute_metrics(bal_scores, bal_labels, bal_thr)
    print(f"[eval] BALANCED TEST (50/50)  F1={metrics_balanced['f1']:.3f}  "
          f"P={metrics_balanced['precision']:.3f}  R={metrics_balanced['recall']:.3f}  "
          f"AUC-PR={metrics_balanced['auc_pr']:.3f}  ROC-AUC={metrics_balanced['roc_auc']:.3f}")

    # ----- plots (Figure 5) ----- #
    plot_paths = evaluate.make_all_plots(test_scores, test_labels, history, cfg.output_dir / "plots")
    print(f"[plots] wrote {len(plot_paths)} figures to {cfg.output_dir / 'plots'}")

    # ----- reference store + autofill data for the demo ----- #
    ref_store = data_prep.compute_reference_store(train_gen, feat)
    demo_aux = data_prep.build_demo_aux(train_raw, test_raw, seed=cfg.seed)

    # ----- save everything ----- #
    artifact = {
        "model_state": model.state_dict(),
        "model_kwargs": model_kwargs,
        "featurizer": asdict(feat),
        "reference_store": ref_store,
        "demo_aux": demo_aux,
        "resid_var": resid_var.astype("float32"),
        "threshold": float(threshold),
        "metrics": metrics,
        "metrics_balanced": metrics_balanced,
        "feature_names": feat.feature_names,
        "recon_names": feat.recon_names,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "is_sample_data": is_sample,
    }
    out_path = cfg.output_dir / ARTIFACT_PATH_NAME
    torch.save(artifact, out_path)
    with open(cfg.output_dir / "metrics.json", "w") as f:
        json.dump({"test_full_imbalanced": metrics, "test_balanced": metrics_balanced,
                   "val_f1": val_f1, "best_val_auc_pr": best_auc,
                   "threshold": threshold, "is_sample_data": is_sample}, f, indent=2)
    print(f"[save] artifacts -> {out_path}")
    print(f"[save] metrics   -> {cfg.output_dir / 'metrics.json'}")

    return {"metrics": metrics, "threshold": threshold, "artifact_path": str(out_path)}
