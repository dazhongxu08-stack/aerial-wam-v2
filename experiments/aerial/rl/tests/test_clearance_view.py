import torch

from experiments.aerial.rl.clearance_view import ClearanceView


def test_side_channels_swap_when_the_picture_flips():
    torch.manual_seed(0)
    model = ClearanceView().eval()
    with torch.no_grad():
        model.side.weight.normal_(0, 0.2)
        model.level.weight.zero_()
        model.level.bias.fill_(1.0)
    rgb = torch.randint(0, 255, (2, 64, 96, 3), dtype=torch.uint8)
    rgb[:, :, :48] = 30
    rgb[:, :, 48:] = 200
    with torch.no_grad():
        straight = model(rgb)
        flipped = model(torch.flip(rgb, dims=(2,)))
    assert torch.allclose(straight[:, 1], flipped[:, 2], atol=1e-4)
    assert torch.allclose(straight[:, 2], flipped[:, 1], atol=1e-4)
