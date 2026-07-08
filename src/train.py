"""Train the hetero-graph auto-encoder ONCE on genuine transactions, then save
everything the demo needs (weights, featuriser, reference store, threshold).

Run via `python main.py` (which calls `run_training`).
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import data_prep, evaluate
from .config import Config
from .model import ae_loss, build_model
from .utils import pick_device, set_seed

ARTIFACT_PATH_NAME = "artifacts.pt"


def _isolated_hetero(sample_df: pd.DataFrame, feats: dict):
    """A batched graph of DISCONNECTED 3-node components (one txn + its own
    customer + merchant), matching how the demo scores a single transaction.
    """
    import torch
    from torch_geometric.data import HeteroData

    X_txn = feats["transaction"].transform(sample_df)
    X_cust = feats["customer"].transform(sample_df)
    X_merch = feats["merchant"].transform(sample_df)
    k = len(sample_df)
    data = HeteroData()
    data["customer"].x = torch.from_numpy(X_cust)
    data["merchant"].x = torch.from_numpy(X_merch)
    data["transaction"].x = torch.from_numpy(X_txn)
    idx = np.arange(k, dtype=np.int64)
    e = np.vstack([idx, idx])
    data["customer", "makes", "transaction"].edge_index = torch.from_numpy(e)
    data["transaction", "rev_makes", "customer"].edge_index = torch.from_numpy(e[[1, 0]])
    data["merchant", "sells", "transaction"].edge_index = torch.from_numpy(e)
    data["transaction", "rev_sells", "merchant"].edge_index = torch.from_numpy(e[[1, 0]])
    return data


def _compute_latent_scatter(model, feats, df: pd.DataFrame, device: str,
                            n: int = 400, fraud_frac: float = 0.3, seed: int = 42) -> dict:
    """Embed a balanced sample of transactions, fit a 2D PCA, and return the
    projected points (+ PCA params so the demo can project new transactions).
    """
    if len(df) == 0:
        return {}
    rng = np.random.default_rng(seed)
    n = min(n, len(df))
    fraud_idx = df.index.values[df["is_fraud"].to_numpy() == 1]
    gen_idx = df.index.values[df["is_fraud"].to_numpy() == 0]
    n_fraud = min(len(fraud_idx), int(round(n * fraud_frac)))
    n_gen = min(len(gen_idx), n - n_fraud)
    pick = np.concatenate([rng.choice(fraud_idx, n_fraud, replace=False),
                           rng.choice(gen_idx, n_gen, replace=False)])
    sample = df.loc[pick]

    emb = model.transaction_embeddings(_isolated_hetero(sample, feats), device)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=seed)
    xy = pca.fit_transform(emb)
    labels = sample["is_fraud"].to_numpy(dtype=int)
    points = [{"x": float(xy[i, 0]), "y": float(xy[i, 1]), "is_fraud": int(labels[i])}
              for i in range(len(sample))]
    return {"points": points, "pca_mean": pca.mean_.tolist(),
            "pca_components": pca.components_.tolist()}


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

    train_path, test_path = data_prep.resolve_data_paths(cfg)
    print(f"[data] train={train_path.name} test={test_path.name}")
    train_raw = data_prep.load_raw(train_path, cfg.train_subsample, cfg.keep_all_fraud, cfg.seed)
    test_raw = data_prep.load_raw(test_path, cfg.test_subsample, cfg.keep_all_fraud, cfg.seed)
    print(f"[data] train rows={len(train_raw)} fraud={int(train_raw.is_fraud.sum())} | "
          f"test rows={len(test_raw)} fraud={int(test_raw.is_fraud.sum())}")

    train_gen, val_df = _split_train(train_raw, cfg.val_fraction, cfg.seed)

    feats = data_prep.make_featurizers()
    for k in feats:
        feats[k].fit(train_gen)                       # genuine-only fit, no leakage
    print(f"[feat] dims: " + ", ".join(f"{t}={feats[t].dim}" for t in feats))

    train_data, _ = data_prep.build_hetero_data(train_gen, feats)
    val_data, val_info = data_prep.build_hetero_data(val_df, feats)
    test_data, test_info = data_prep.build_hetero_data(test_raw, feats)

    metadata = data_prep.HETERO_METADATA
    in_dims = {t: feats[t].dim for t in feats}
    type_targets = data_prep.type_targets_from_featurizers(feats)

    import torch
    train_data = train_data.to(device)
    counts = {t: train_data[t].x.size(0) for t in in_dims}
    total_nodes = sum(counts.values())
    prevalence = {t: counts[t] / total_nodes for t in counts}   # Eq. 7 prevalence weights

    model = build_model(cfg, metadata, in_dims, type_targets).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = None
    if cfg.use_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        sched = ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                  patience=max(3, cfg.early_stop_patience // 3))
    tricks = [f"denoise={cfg.denoise_std}" if cfg.denoise_std > 0 else None,
              "scheduler" if cfg.use_scheduler else None,
              f"grad_clip={cfg.grad_clip}" if cfg.grad_clip > 0 else None,
              "layernorm" if cfg.use_layernorm else None]
    tricks = [t for t in tricks if t]
    print(f"[train] layers={cfg.encoder_layers} dropout={cfg.dropout} "
          f"tricks=[{', '.join(tricks) or 'none (paper-faithful)'}]")

    val_labels = val_info["labels"]
    history: dict[str, list] = {}
    best_auc, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        # denoising: reconstruct the CLEAN inputs from noise-corrupted ones
        x_in = train_data.x_dict
        if cfg.denoise_std > 0:
            x_in = {t: v + torch.randn_like(v) * cfg.denoise_std
                    for t, v in train_data.x_dict.items()}
        recon, mu, logvar = model(x_in, train_data.edge_index_dict)
        loss, parts = ae_loss(recon, train_data.x_dict, type_targets, prevalence)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        val_scores = model.transaction_scores(val_data, device).numpy()
        # per-epoch diagnostics (threshold, genuine/fraud error gap, val metrics,
        # per-type loss) so the training process can be visualized after the fact
        diag = evaluate.epoch_diagnostics(val_scores, val_labels, parts)
        diag["train_loss"] = float(loss.item())
        diag["val_loss"] = diag["val_err_genuine_mean"]   # kept for backward-compat
        for k, v in diag.items():
            history.setdefault(k, []).append(v)
        val_auc = diag["val_auc_pr"]
        if sched is not None and np.isfinite(val_auc):
            sched.step(val_auc)

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"[epoch {epoch:3d}] loss={loss.item():.5f} val_auc_pr={val_auc:.4f}")
        if patience >= cfg.early_stop_patience:
            print(f"[early stop] no val improvement for {cfg.early_stop_patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- thresholds: mu+2sigma (decision rule) + F1-sweep (comparison) ----- #
    val_scores = model.transaction_scores(val_data, device).numpy()
    genuine_val = val_scores[val_labels == 0]
    threshold_mu2sigma = evaluate.mu_2sigma_threshold(genuine_val)
    threshold_f1, val_f1 = evaluate.search_threshold(val_scores, val_labels)

    test_scores = model.transaction_scores(test_data, device).numpy()
    test_labels = test_info["labels"]
    metrics = evaluate.compute_metrics(test_scores, test_labels, threshold_mu2sigma)
    metrics_f1 = evaluate.compute_metrics(test_scores, test_labels, threshold_f1)
    print(f"[eval] mu+2sigma thr={threshold_mu2sigma:.6f}  "
          f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
          f"AUC-PR={metrics['auc_pr']:.3f} ROC-AUC={metrics['roc_auc']:.3f}")
    print(f"[eval] F1-sweep thr={threshold_f1:.6f} (val F1={val_f1:.3f})  "
          f"test F1={metrics_f1['f1']:.3f}")

    plot_paths = evaluate.make_all_plots(test_scores, test_labels, history, cfg.output_dir / "plots")
    print(f"[plots] wrote {len(plot_paths)} figures")

    ref_store = data_prep.compute_reference_store(train_gen, feats)
    demo_aux = data_prep.build_demo_aux(train_raw, test_raw, seed=cfg.seed)
    viz_graph = data_prep.build_viz_graph(train_raw, cfg.viz_max_nodes, cfg.seed,
                                          cfg.viz_fraud_frac)
    latent_scatter = _compute_latent_scatter(model, feats, train_raw, device,
                                             seed=cfg.seed, fraud_frac=cfg.viz_fraud_frac)
    # EDA panel: profile the FULL files (4 columns only), not the training subsample
    data_profile = data_prep.compute_data_profile(
        data_prep.load_profile_frame(train_path), data_prep.load_profile_frame(test_path))

    model_kwargs = dict(metadata=metadata, in_dims=in_dims, type_targets=type_targets,
                        hidden_dim=cfg.hidden_dim, heads=cfg.heads,
                        encoder_layers=cfg.encoder_layers, latent_dim=cfg.latent_dim,
                        decoder_hidden=cfg.decoder_hidden, dropout=cfg.dropout,
                        use_layernorm=cfg.use_layernorm)
    artifact = {
        "model_state": model.state_dict(),
        "model_kwargs": model_kwargs,
        "featurizers": {k: asdict(f) for k, f in feats.items()},
        "reference_store": ref_store,
        "demo_aux": demo_aux,
        "viz_graph": viz_graph,
        "latent_scatter": latent_scatter,
        "prevalence": prevalence,
        "threshold_mu2sigma": float(threshold_mu2sigma),
        "threshold_f1": float(threshold_f1),
        "metrics": metrics,
        "metrics_f1": metrics_f1,
        "feature_names": {t: feats[t].feature_names for t in feats},
        "history": history,          # per-epoch train/val curves for the demo
        "data_profile": data_profile,  # EDA summary (class balance, amount/hour/category) for the demo
    }
    out_path = cfg.output_dir / ARTIFACT_PATH_NAME
    torch.save(artifact, out_path)
    if history:
        hist_df = pd.DataFrame(history)
        hist_df.index = np.arange(1, len(hist_df) + 1)
        hist_df.to_csv(cfg.output_dir / "history.csv", index_label="epoch")
        print(f"[save] per-epoch history -> {cfg.output_dir / 'history.csv'}")
    with open(cfg.output_dir / "metrics.json", "w") as f:
        json.dump({"test_mu2sigma": metrics, "test_f1sweep": metrics_f1,
                   "threshold_mu2sigma": threshold_mu2sigma, "threshold_f1": threshold_f1,
                   "best_val_auc_pr": best_auc}, f, indent=2)
    print(f"[save] artifacts -> {out_path}")
    return {"metrics": metrics, "threshold": threshold_mu2sigma, "artifact_path": str(out_path)}
