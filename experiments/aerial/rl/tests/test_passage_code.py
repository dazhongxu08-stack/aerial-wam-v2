"""Street code does not start the passage. Wall code plays the recorded sticks."""
import numpy as np
import torch

from experiments.aerial.rl.passage_code import PassageEncoder, PassagePlayer


def test_street_code_stays_with_the_actor_and_wall_code_plays_the_tape():
    encoder = PassageEncoder()
    with torch.no_grad():
        encoder.street.copy_(torch.tensor([1.0] + [0.0] * 31))
        encoder.wall.copy_(torch.tensor([0.0, 1.0] + [0.0] * 30))
    actions = np.array([[0.0, 0.0, 0.0, -0.3], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    embeds = np.zeros((3, 32), dtype=np.float32)
    embeds[0, 1] = 1.0
    player = PassagePlayer(encoder, embeds, actions, release_idx=2, margin=0.05)

    def fake_embed(rgb):
        z = torch.zeros(1, 32)
        z[0, int(rgb.reshape(-1)[0].item())] = 1.0
        return z

    encoder.embed = fake_embed  # type: ignore[method-assign]
    assert player.maybe(0) is None  # street axis
    first = player.maybe(1)  # wall axis
    assert first is not None and abs(first[3] + 0.3) < 1e-6
    second = player.maybe(0)  # picture no longer matters once the tape is running
    assert second is not None and abs(second[0] - 1.0) < 1e-6
    assert player.maybe(1) is None  # released back to the actor
