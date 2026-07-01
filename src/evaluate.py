"""Threshold search, metrics and the plots from the paper (Figure 5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# headless backend so plotting works without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def search_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find the reconstruction-error threshold maximising F1 (paper Fig. 5c)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float(np.median(scores)), 0.0
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    # thresholds has len-1 vs precision/recall; align by dropping last p/r point
    best = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    thr = float(thresholds[best]) if len(thresholds) else float(np.median(scores))
    return thr, float(f1[best])


def mu_2sigma_threshold(genuine_scores) -> float:
    """Paper Eq. 9: Threshold = mu + 2*sigma of genuine reconstruction errors."""
    s = np.asarray(genuine_scores, dtype=float)
    return float(s.mean() + 2.0 * s.std())


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (scores >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "accuracy": float(accuracy_score(labels, pred)),
        "n": int(len(labels)),
        "n_fraud": int(labels.sum()),
    }
    if labels.sum() > 0 and labels.sum() < len(labels):
        out["roc_auc"] = float(roc_auc_score(labels, scores))
        out["auc_pr"] = float(average_precision_score(labels, scores))
    else:
        out["roc_auc"] = float("nan")
        out["auc_pr"] = float("nan")
    return out


def epoch_diagnostics(val_scores, val_labels, loss_parts: dict | None = None) -> dict:
    """Per-epoch training diagnostics for visualizing the training process.

    Everything here is derived from the VALIDATION split (never the test set —
    peeking at test each epoch would leak it). Returns a flat dict of scalars so
    the caller can append each value into a per-series history:

      * threshold (paper Eq. 9): ``thr_mu``, ``thr_sigma``, ``thr_mu2sigma`` on
        held-out genuine reconstruction errors;
      * error gap: mean/median recon error for genuine vs fraud rows — the
        separation the anomaly detector relies on;
      * metrics @ that epoch's mu+2sigma threshold: ``val_precision``,
        ``val_recall``, ``val_f1``, plus ``val_roc_auc`` / ``val_auc_pr``;
      * per-type reconstruction loss (from ``ae_loss``'s ``parts``), if given.
    """
    scores = np.asarray(val_scores, dtype=float)
    labels = np.asarray(val_labels, dtype=int)
    genuine = scores[labels == 0]
    fraud = scores[labels == 1]

    mu = float(genuine.mean()) if genuine.size else float("nan")
    sigma = float(genuine.std()) if genuine.size else float("nan")
    thr = mu + 2.0 * sigma

    out = {
        "thr_mu": mu,
        "thr_sigma": sigma,
        "thr_mu2sigma": thr,
        "val_err_genuine_mean": mu,
        "val_err_fraud_mean": float(fraud.mean()) if fraud.size else float("nan"),
        "val_err_genuine_median": float(np.median(genuine)) if genuine.size else float("nan"),
        "val_err_fraud_median": float(np.median(fraud)) if fraud.size else float("nan"),
    }

    m = compute_metrics(scores, labels, thr)
    out["val_precision"] = m["precision"]
    out["val_recall"] = m["recall"]
    out["val_f1"] = m["f1"]
    out["val_roc_auc"] = m["roc_auc"]
    out["val_auc_pr"] = m["auc_pr"]

    if loss_parts:
        for t, v in loss_parts.items():
            out[f"loss_{t}"] = float(v.detach()) if hasattr(v, "detach") else float(v)
    return out


# --------------------------------------------------------------------------- #
# plots (mirror Figure 5 a-e)
# --------------------------------------------------------------------------- #
def plot_loss_curves(history: dict, out: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Training loss")
    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def plot_score_distribution(scores, labels, out: Path) -> None:
    scores = np.asarray(scores); labels = np.asarray(labels)
    plt.figure(figsize=(6, 4))
    neg, pos = scores[labels == 0], scores[labels == 1]
    bins = np.linspace(scores.min(), np.percentile(scores, 99.5), 60)
    plt.hist(neg, bins=bins, alpha=0.6, label="Negative (genuine)")
    plt.hist(pos, bins=bins, alpha=0.6, label="Positive (fraud)")
    plt.xlabel("Reconstruction error"); plt.ylabel("Frequency")
    plt.title("Distribution of Reconstruction Error"); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def plot_f1_vs_threshold(scores, labels, out: Path) -> None:
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    precision, recall, thr = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    plt.figure(figsize=(6, 4))
    plt.plot(thr, f1[:-1])
    plt.xlabel("Threshold value"); plt.ylabel("F1 score")
    plt.title("F1 score vs. Threshold")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def plot_roc(scores, labels, out: Path) -> None:
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def plot_pr(scores, labels, out: Path) -> None:
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"Proposed Model (AUC-PR = {ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def make_all_plots(scores, labels, history, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    jobs = [
        ("loss_curves.png", lambda p: plot_loss_curves(history, p)),
        ("score_distribution.png", lambda p: plot_score_distribution(scores, labels, p)),
        ("f1_vs_threshold.png", lambda p: plot_f1_vs_threshold(scores, labels, p)),
        ("roc_curve.png", lambda p: plot_roc(scores, labels, p)),
        ("pr_curve.png", lambda p: plot_pr(scores, labels, p)),
    ]
    for name, fn in jobs:
        p = out_dir / name
        try:
            fn(p); paths.append(p)
        except Exception as e:  # a degenerate split shouldn't crash training
            print(f"[plot] skipped {name}: {e}")
    return paths
