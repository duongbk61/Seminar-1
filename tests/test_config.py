from src.config import get_config


def test_faithful_hyperparameters():
    c = get_config()
    assert c.hidden_dim == 64
    assert c.heads == 16
    assert c.decoder_hidden == 64
    assert c.dropout == 0.4
    assert c.weight_decay == 0.01
    assert c.beta == 0.0          # no KL term
    assert c.latent_dim == 64     # no bottleneck
    assert c.encoder_layers == 2  # forced deviation from Table 3's 124
    assert c.hidden_dim % c.heads == 0
