import torch

from nbs_pg.box_geometry import BoxHierarchyCalibrator, box_pair_metrics, build_go_box_edge_features


def test_exact_and_soft_inclusion():
    child_c = torch.tensor([[0.0, 0.0]])
    child_o = torch.tensor([[0.2, 0.2]])
    parent_c = torch.tensor([[0.0, 0.0]])
    parent_o = torch.tensor([[0.5, 0.5]])
    m = box_pair_metrics(child_c, child_o, parent_c, parent_o)
    assert m.worst_margin.item() > 0
    assert m.normalized_violation.item() == 0

    bad_parent_o = torch.tensor([[0.1, 0.1]])
    m2 = box_pair_metrics(child_c, child_o, parent_c, bad_parent_o)
    assert m2.worst_margin.item() < 0
    assert m2.normalized_violation.item() > 0


def test_edge_feature_width_and_calibration():
    center = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    offset = torch.tensor([[0.5, 0.5], [0.2, 0.2], [0.1, 0.1]])
    edge = torch.tensor([[1, 2], [0, 0]])
    cal = BoxHierarchyCalibrator(-0.15, 10.0, trainable=True)
    feat = build_go_box_edge_features(center, offset, edge, calibrator=cal)
    assert feat.shape == (2, 8)
    assert torch.all((feat[:, 0] >= 0) & (feat[:, 0] <= 1))
