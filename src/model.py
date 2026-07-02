"""Heterogeneous Graph Auto-Encoder (paper Section 4), faithful implementation.

Encoder : per-type input projection -> stack of custom HGAEConv layers (intra-edge
          attention, Eq. 1-5). After each layer, a VAE-style reparameterization is
          applied per node type (Algorithm 1, line 7).
Decoder : per-type MLP reconstructing each node type's FULL attribute vector -
          continuous features (MSE) and categorical features (cross-entropy),
          Eq. 6-8.
Score   : a transaction's combined reconstruction error (anomaly score).

Two forced deviations from a literal reading (see src/config.py):
  * encoder_layers small (Table 3's 124 oversmooths);
  * standard mu/logvar reparameterization (the literal log(h) formula NaNs).
There is NO KL term (the paper's loss is reconstruction-only).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .hgae_conv import HGAEConv


class HeteroGraphAutoEncoder(nn.Module):
    def __init__(self, metadata, in_dims, type_targets, hidden_dim=64, heads=16,
                 encoder_layers=2, latent_dim=64, decoder_hidden=64, dropout=0.4,
                 use_layernorm=False):
        super().__init__()
        self.metadata = metadata
        self.node_types = list(metadata[0])
        self.type_targets = type_targets
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        self.in_lin = nn.ModuleDict({t: nn.Linear(in_dims[t], hidden_dim) for t in self.node_types})
        self.convs = nn.ModuleList([HGAEConv(metadata, hidden_dim, heads) for _ in range(encoder_layers)])
        # optional LayerNorm per (layer, node type) to stabilize + fight oversmoothing
        self.norms = nn.ModuleList([
            nn.ModuleDict({t: nn.LayerNorm(hidden_dim) for t in self.node_types})
            for _ in range(encoder_layers)
        ]) if use_layernorm else None
        self.fc_mu = nn.ModuleDict({t: nn.Linear(hidden_dim, latent_dim) for t in self.node_types})
        self.fc_logvar = nn.ModuleDict({t: nn.Linear(hidden_dim, latent_dim) for t in self.node_types})

        # per-type decoder: shared trunk -> continuous head + one head per categorical group
        self.dec_trunk = nn.ModuleDict()
        self.dec_cont = nn.ModuleDict()
        self.dec_cat = nn.ModuleDict()
        for t in self.node_types:
            self.dec_trunk[t] = nn.Sequential(
                nn.Linear(latent_dim, decoder_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(decoder_hidden, decoder_hidden), nn.ReLU(), nn.Dropout(dropout),
            )
            self.dec_cont[t] = nn.Linear(decoder_hidden, max(1, type_targets[t]["cont"]))
            self.dec_cat[t] = nn.ModuleDict(
                {name: nn.Linear(decoder_hidden, width)
                 for (name, _start, width) in type_targets[t]["cat_groups"]}
            )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def encode(self, x_dict, edge_index_dict):
        h = {t: F.relu(self.in_lin[t](x)) for t, x in x_dict.items()}
        mu = logvar = None
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict)
            if self.norms is not None:
                h = {t: self.norms[i][t](v) for t, v in h.items()}
            h = {t: F.dropout(F.relu(v), p=self.dropout, training=self.training) for t, v in h.items()}
            mu = {t: self.fc_mu[t](h[t]) for t in h}
            logvar = {t: self.fc_logvar[t](h[t]) for t in h}
            h = {t: self.reparameterize(mu[t], logvar[t]) for t in h}
        return h, mu, logvar

    def forward(self, x_dict, edge_index_dict):
        z, mu, logvar = self.encode(x_dict, edge_index_dict)
        recon = {}
        for t in z:
            trunk = self.dec_trunk[t](z[t])
            recon[t] = {
                "cont": self.dec_cont[t](trunk),
                "cats": {name: head(trunk) for name, head in self.dec_cat[t].items()},
            }
        return recon, mu, logvar

    @torch.no_grad()
    def transaction_scores(self, data, device="cpu") -> torch.Tensor:
        """Per-transaction combined reconstruction error (anomaly score)."""
        self.eval()
        data = data.to(device)
        recon, _, _ = self.forward(data.x_dict, data.edge_index_dict)
        t = "transaction"
        x = data[t].x
        tgt = self.type_targets[t]
        n_cont = tgt["cont"]
        err = ((recon[t]["cont"] - x[:, :n_cont]) ** 2).mean(dim=1)
        for (name, start, width) in tgt["cat_groups"]:
            cls = x[:, start:start + width].argmax(dim=1)
            err = err + F.cross_entropy(recon[t]["cats"][name], cls, reduction="none")
        return err.cpu()

    @torch.no_grad()
    def transaction_feature_errors(self, data, device="cpu") -> dict:
        """Per continuous-feature squared error for transactions (demo 'why')."""
        self.eval()
        data = data.to(device)
        recon, _, _ = self.forward(data.x_dict, data.edge_index_dict)
        t = "transaction"
        x = data[t].x
        n_cont = self.type_targets[t]["cont"]
        sq = (recon[t]["cont"] - x[:, :n_cont]) ** 2
        return {i: sq[:, i].cpu() for i in range(n_cont)}

    @torch.no_grad()
    def transaction_embeddings(self, data, device="cpu"):
        """Latent embeddings (mu) for transaction nodes — powers the 2D scatter."""
        self.eval()
        data = data.to(device)
        _, mu, _ = self.encode(data.x_dict, data.edge_index_dict)
        return mu["transaction"].cpu().numpy()

    @torch.no_grad()
    def transaction_reconstruction(self, data, device="cpu") -> dict:
        """Per continuous-feature (reconstructed, actual, squared-error) for txns.

        Powers the demo's 'expected vs actual' panel: a genuine transaction is
        reconstructed close to its input, a fraud diverges — that gap IS the
        anomaly signal.
        """
        self.eval()
        data = data.to(device)
        recon, _, _ = self.forward(data.x_dict, data.edge_index_dict)
        t = "transaction"
        x = data[t].x
        n_cont = self.type_targets[t]["cont"]
        rec = recon[t]["cont"]
        sq = (rec - x[:, :n_cont]) ** 2
        return {i: (rec[:, i].cpu(), x[:, i].cpu(), sq[:, i].cpu()) for i in range(n_cont)}


def ae_loss(recon, x_dict, type_targets, prevalence):
    """Reconstruction loss: MSE(continuous) + CE(categorical), prevalence-weighted.

    NO KL term (the paper's objective is reconstruction-only).
    """
    total = 0.0
    parts = {}
    for t, tgt in type_targets.items():
        x = x_dict[t]
        n_cont = tgt["cont"]
        l = F.mse_loss(recon[t]["cont"], x[:, :n_cont])
        for (name, start, width) in tgt["cat_groups"]:
            cls = x[:, start:start + width].argmax(dim=1)
            l = l + F.cross_entropy(recon[t]["cats"][name], cls)
        parts[t] = l
        total = total + prevalence[t] * l
    return total, parts


def build_model(cfg, metadata, in_dims, type_targets) -> HeteroGraphAutoEncoder:
    return HeteroGraphAutoEncoder(
        metadata=metadata, in_dims=in_dims, type_targets=type_targets,
        hidden_dim=cfg.hidden_dim, heads=cfg.heads, encoder_layers=cfg.encoder_layers,
        latent_dim=cfg.latent_dim, decoder_hidden=cfg.decoder_hidden, dropout=cfg.dropout,
        use_layernorm=getattr(cfg, "use_layernorm", False),
    )
