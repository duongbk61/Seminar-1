"""Custom intra-edge attention layer for the HGAE encoder (paper Eq. 1-5).

Distinctive to this paper (vs. standard HGT / GAT): the softmax is normalized
ACROSS THE H ATTENTION HEADS of a single edge (`intra_edge_softmax`), and
neighbour messages are aggregated by a UNIFORM SUM (element-wise addition, the
paper's oplus), NOT a neighbour-normalized weighting. Each relation r owns its
learnable W_Att_r and W_Mssg_r; each (node type, head) owns source/dest/message
projections.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _rk(rel) -> str:
    return "__".join(rel)


def intra_edge_softmax(att: torch.Tensor) -> torch.Tensor:
    """Softmax across the H heads of each edge. att: [E, H] -> [E, H]."""
    return torch.softmax(att, dim=1)


class HGAEConv(nn.Module):
    def __init__(self, metadata, hidden_dim: int = 64, heads: int = 16) -> None:
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.node_types, self.edge_types = metadata[0], metadata[1]
        self.hidden = hidden_dim
        self.heads = heads
        self.dk = hidden_dim // heads

        # per-(type) projections producing all heads at once (view to [N,H,dk])
        self.s_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.node_types})
        self.d_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.node_types})
        self.m_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.node_types})
        # Eq. 1 output projection per destination type (no bias -> testable scaling)
        self.out_lin = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim, bias=False) for t in self.node_types})

        # per-relation learnable matrices W_Att_r, W_Mssg_r : [H, dk, dk]
        self.w_att = nn.ParameterDict()
        self.w_mssg = nn.ParameterDict()
        for rel in self.edge_types:
            wa = torch.empty(heads, self.dk, self.dk)
            wm = torch.empty(heads, self.dk, self.dk)
            nn.init.xavier_uniform_(wa)
            nn.init.xavier_uniform_(wm)
            self.w_att[_rk(rel)] = nn.Parameter(wa)
            self.w_mssg[_rk(rel)] = nn.Parameter(wm)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        H, dk = self.heads, self.dk
        # aggregation accumulators per destination type
        agg = {t: x_dict[t].new_zeros(x_dict[t].size(0), H, dk) for t in x_dict}

        for rel, edge_index in edge_index_dict.items():
            src_t, _, dst_t = rel
            src_idx, dst_idx = edge_index[0], edge_index[1]
            s = self.s_proj[src_t](x_dict[src_t]).view(-1, H, dk)[src_idx]   # [E,H,dk]
            d = self.d_proj[dst_t](x_dict[dst_t]).view(-1, H, dk)[dst_idx]   # [E,H,dk]
            m = self.m_proj[src_t](x_dict[src_t]).view(-1, H, dk)[src_idx]   # [E,H,dk]

            wa = self.w_att[_rk(rel)]                                        # [H,dk,dk]
            wm = self.w_mssg[_rk(rel)]                                       # [H,dk,dk]
            att = torch.einsum("ehi,hij,ehj->eh", s, wa, d) / math.sqrt(dk)  # [E,H]
            alpha = intra_edge_softmax(att)                                  # [E,H]
            msg = torch.einsum("ehi,hij->ehj", m, wm)                        # [E,H,dk]
            contrib = alpha.unsqueeze(-1) * msg                             # [E,H,dk]
            agg[dst_t].index_add_(0, dst_idx, contrib)                      # uniform sum

        out = {}
        for t in x_dict:
            flat = agg[t].reshape(agg[t].size(0), H * dk)
            out[t] = self.out_lin[t](flat) + x_dict[t]                       # Eq. 1 residual
        return out
