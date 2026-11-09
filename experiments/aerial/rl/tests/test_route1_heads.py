"""The extra route-1 heads stay inside the body box."""
import torch

from experiments.aerial.rl.action_chunk import ActionChunkNet
from experiments.aerial.rl.clearance_head import ClearanceHead


def test_clearance_and_chunk_shapes():
    head = ClearanceHead()
    feat = torch.zeros(2, 9)
    out = head(feat)
    assert tuple(out.shape) == (2, 4)
    chunk = ActionChunkNet(horizon=8)
    rgb = torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8)
    seq = chunk(rgb)
    assert tuple(seq.shape) == (1, 8, 4)
