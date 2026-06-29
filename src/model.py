"""Heterogeneous Graph Auto-Encoder with attention (paper Section 4).

Encoder  : per-type input projection -> stack of HGT layers (multi-head,
           edge-type-specific attention + message passing, Eq. 2-5). HGTConv is
           PyG's implementation of Heterogeneous Graph Transformer (Hu et al.
           2020), which is reference [41] the paper builds on.
Latent   : variational reparameterisation on the transaction embeddings
           (Eq. at end of 4.1): z = mu + eps * exp(0.5 * logvar).
Decoder  : an MLP (deep NN, Eq. 6) that reconstructs the transaction feature
           vector from z. Anomaly score = per-transaction reconstruction error.

We deviate from a literal reading of Table 4 ("124 encoder layers") and apply
the VAE reparameterisation once on the final transaction embedding rather than
inside every layer; both are standard and keep the model trainable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HGTConv


class HeteroGraphAutoEncoder(nn.Module):
    def __init__(
        self,
        metadata,
        in_dims: dict[str, int],
        recon_idx: list[int],
        hidden_dim: int = 64,
        heads: int = 8,
        encoder_layers: int = 2,
        latent_dim: int = 64,
        decoder_hidden: int = 64,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.node_types = list(metadata[0])
        self.dropout = dropout
        self.recon_dim = len(recon_idx)
        self.register_buffer("recon_idx", torch.as_tensor(recon_idx, dtype=torch.long))

        self.in_lin = nn.ModuleDict(
            {nt: nn.Linear(in_dims[nt], hidden_dim) for nt in self.node_types}
        )
        self.convs = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, metadata, heads=heads) for _ in range(encoder_layers)]
        )

        # VAE heads (applied to the transaction embedding only)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # MLP decoder -> reconstruct the original transaction feature vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, self.recon_dim),
        )

    # --- encoder ---------------------------------------------------------- #
    def encode(self, x_dict, edge_index_dict):
        h = {nt: F.relu(self.in_lin[nt](x)) for nt, x in x_dict.items()}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {
                nt: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                for nt, v in h.items()
            }
        z = h["transaction"]
        return self.fc_mu(z), self.fc_logvar(z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu  # deterministic at eval -> stable reconstruction error

    def forward(self, x_dict, edge_index_dict):
        mu, logvar = self.encode(x_dict, edge_index_dict)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    # --- scoring ---------------------------------------------------------- #
    @torch.no_grad()
    def reconstruction_error(self, data, device="cpu") -> torch.Tensor:
        """Per-transaction mean-squared reconstruction error (the anomaly score)."""
        self.eval()
        data = data.to(device)
        recon, _, _ = self.forward(data.x_dict, data.edge_index_dict)
        target = data["transaction"].x[:, self.recon_idx]
        return F.mse_loss(recon, target, reduction="none").mean(dim=1).cpu()

    @torch.no_grad()
    def per_feature_error(self, data, device="cpu") -> torch.Tensor:
        """Squared error per feature per transaction (for demo explanations)."""
        self.eval()
        data = data.to(device)
        recon, _, _ = self.forward(data.x_dict, data.edge_index_dict)
        target = data["transaction"].x[:, self.recon_idx]
        return ((recon - target) ** 2).cpu()

    @torch.no_grad()
    def anomaly_scores(self, data, resid_var=None, device="cpu") -> torch.Tensor:
        """Per-transaction anomaly score.

        If `resid_var` (per-feature genuine residual variance) is given, the score
        is the mean *standardised* squared error - features the model reconstructs
        well for genuine traffic (e.g. amount) dominate, noisy features (e.g. the
        random category) are down-weighted. This is what separates fraud cleanly.
        """
        pfe = self.per_feature_error(data, device)
        if resid_var is not None:
            pfe = pfe / resid_var
        return pfe.mean(dim=1)


def ae_loss(recon, target, mu, logvar, beta: float):
    """VAE objective: reconstruction MSE + beta * KL, plus per-node recon error."""
    per_node = F.mse_loss(recon, target, reduction="none").mean(dim=1)
    recon_loss = per_node.mean()
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta * kl, recon_loss, kl


def build_model(cfg, metadata, in_dims, recon_idx) -> HeteroGraphAutoEncoder:
    return HeteroGraphAutoEncoder(
        metadata=metadata,
        in_dims=in_dims,
        recon_idx=recon_idx,
        hidden_dim=cfg.hidden_dim,
        heads=cfg.heads,
        encoder_layers=cfg.encoder_layers,
        latent_dim=cfg.latent_dim,
        decoder_hidden=cfg.decoder_hidden,
        dropout=cfg.dropout,
    )
