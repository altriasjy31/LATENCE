import torch

from nbs_pg.config import NBSConfig
from nbs_pg.go_encoder import BoxSquaredGOEncoder


def test_center_offset_are_separate_and_fused():
    torch.manual_seed(3)
    cfg = NBSConfig(hidden_dim=16, go_stat_dim=6, go_tower_layers=0)
    enc = BoxSquaredGOEncoder(cfg, box_dim=8)
    center = torch.randn(7, 8)
    offset = torch.rand(7, 8) + 0.01
    stats = torch.randn(7, 6)
    out = enc(center, offset, stats)
    assert out.semantic.shape == (7, 16)
    assert out.hierarchy.shape == (7, 16)
    assert out.static.shape == (7, 16)
    assert out.hierarchy_gate.shape == (7, 16)
    assert torch.isfinite(out.static).all()
