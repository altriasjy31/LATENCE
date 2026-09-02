import torch


def test_factorized_context_matches_explicit():
    torch.manual_seed(1)
    b, s, n, d = 3, 4, 11, 7
    q = torch.randn(b, d)
    c = torch.randn(n, d)
    sources = torch.randn(s, n, d)
    weights = torch.softmax(torch.randn(b, s), dim=1)
    gates = torch.sigmoid(torch.randn(b, s, d))
    scale = torch.tensor(0.37)

    explicit_context = c.unsqueeze(0) + scale * torch.einsum(
        "bs,bsd,snd->bnd", weights, gates, sources
    )
    explicit = torch.einsum("bd,bnd->bn", q, explicit_context)
    routed_query = weights.unsqueeze(-1) * gates * q.unsqueeze(1)
    factorized = torch.einsum("bd,nd->bn", q, c) + scale * torch.einsum(
        "bsd,snd->bn", routed_query, sources
    )
    torch.testing.assert_close(explicit, factorized)
