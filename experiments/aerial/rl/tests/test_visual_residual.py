"""The image residual stays inside the body box."""
import torch

from experiments.aerial.rl.visual_residual import VisualResidual


def test_residual_matches_action_box():
    model = VisualResidual()
    rgb = torch.randint(0, 255, (2, 224, 224, 3), dtype=torch.uint8)
    out = model(rgb)
    assert tuple(out.shape) == (2, 4)
    limits = model.limits
    assert torch.all(out.abs() <= limits + 1e-5)
