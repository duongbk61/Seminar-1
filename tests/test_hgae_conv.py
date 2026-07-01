import torch

from src.data_prep import HETERO_METADATA
from src.hgae_conv import HGAEConv, intra_edge_softmax


def test_intra_edge_softmax_sums_to_one_across_heads():
    att = torch.randn(5, 4)  # 5 edges, 4 heads
    alpha = intra_edge_softmax(att)
    assert alpha.shape == (5, 4)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(5), atol=1e-5)


def _two_node_graph(n_customers):
    # n_customers identical customers all pointing at ONE transaction.
    # Merchant edges are intentionally omitted so that the only incoming
    # relation for the transaction node is customer->transaction; this makes
    # the 2x uniform-sum test exact (no constant merchant contribution).
    x_dict = {
        "customer": torch.ones(n_customers, 8),
        "merchant": torch.ones(1, 8),
        "transaction": torch.zeros(1, 8),
    }
    c2t = torch.tensor([[i for i in range(n_customers)], [0] * n_customers])
    m2t = torch.zeros(2, 0, dtype=torch.long)  # no merchant edges in this test graph
    edge_index_dict = {
        ("customer", "makes", "transaction"): c2t,
        ("transaction", "rev_makes", "customer"): c2t[[1, 0]],
        ("merchant", "sells", "transaction"): m2t,
        ("transaction", "rev_sells", "merchant"): m2t,
    }
    return x_dict, edge_index_dict


def test_uniform_neighbour_aggregation_is_a_sum():
    torch.manual_seed(0)
    conv = HGAEConv(HETERO_METADATA, hidden_dim=8, heads=2)
    conv.eval()
    x1, e1 = _two_node_graph(1)
    x2, e2 = _two_node_graph(2)  # duplicate the identical customer neighbour
    with torch.no_grad():
        out1 = conv(x1, e1)["transaction"]
        out2 = conv(x2, e2)["transaction"]
    # residual is x_dict["transaction"] (zeros) in both; out_lin has no bias, so
    # doubling identical neighbours doubles the transaction output.
    assert torch.allclose(out2, 2 * out1, atol=1e-5)


def test_forward_output_shapes():
    conv = HGAEConv(HETERO_METADATA, hidden_dim=8, heads=2)
    x, e = _two_node_graph(3)
    out = conv(x, e)
    assert out["transaction"].shape == (1, 8)
    assert out["customer"].shape == (3, 8)
    assert out["merchant"].shape == (1, 8)
