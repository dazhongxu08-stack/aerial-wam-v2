"""G0 loss: multi-step occupancy on the prior, plus the existing KL.

Pixel, reward, depth, and collision heads are not in this graph. The tensors
that enter the world model are RGB and the 10-D flight proprio. Depth and pose
are used only to build ``occ``, before this function is called.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.aerial.rl.dynamics_torch import _categorical_kl
from experiments.aerial.rl.occ_grid import N_CELLS, N_CLASSES

HORIZON = 8


class OccDecoder(nn.Module):
    """1536 → 256 → 7200×3."""

    def __init__(self, in_dim: int = 1536, hidden: int = 256) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), N_CELLS * N_CLASSES),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        flat = feature.reshape(-1, feature.shape[-1])
        logits = self.net(flat).reshape(*feature.shape[:-1], N_CELLS, N_CLASSES)
        return logits


def class_balanced_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum of per-class means, averaged over the batch. Empty classes are skipped."""
    logp = logits.reshape(logits.shape[0], -1, N_CLASSES)
    tgt = target.reshape(target.shape[0], -1).long()
    losses = []
    for b in range(int(logp.shape[0])):
        total = logp.new_zeros(())
        used = 0
        for cls in range(N_CLASSES):
            mask = tgt[b] == cls
            if not bool(mask.any()):
                continue
            total = total + F.cross_entropy(logp[b][mask], tgt[b][mask])
            used += 1
        if used == 0:
            total = logp[b].sum() * 0.0
        losses.append(total)
    return torch.stack(losses).mean()


def occ_topo_loss(
    model: torch.nn.Module,
    decoder: OccDecoder,
    rgb: torch.Tensor,
    proprio: torch.Tensor,
    action: torch.Tensor,
    occ: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Occupancy cross-entropy plus KL.

    ``rgb`` is ``[B, H, h, w, 3]`` uint8. ``proprio`` is ``[B, H, 10]``.
    ``action`` is ``[B, H + K - 1, 4]``: index ``t`` is the action after frame
    ``t``. The anchor is frame ``H - 1``. ``occ`` is ``[B, 1 + K, NZ, NY, NX]``
    with index 0 the posterior label and indices ``1..K`` the action-integrated
    prior labels. ``K`` is 8.
    """
    if int(proprio.shape[-1]) != int(model.proprio_dim):
        raise ValueError(
            "proprio width %s does not match the model (%s)"
            % (int(proprio.shape[-1]), int(model.proprio_dim))
        )
    if int(model.proprio_dim) != 10:
        raise ValueError("occ_topo requires proprio_dim 10, got %s" % model.proprio_dim)
    b, hist = int(proprio.shape[0]), int(proprio.shape[1])
    k_horizon = int(occ.shape[1]) - 1
    if k_horizon != HORIZON:
        raise ValueError("expected %s prior steps, got %s" % (HORIZON, k_horizon))

    h = model.rssm.initial_h(b, model.device, model.torch_dtype)
    post_seq = []
    prior_seq = []
    anchor_feat = None
    for t in range(hist):
        embed = model._embed(rgb[:, t], proprio[:, t])
        prior_p = model.rssm.prior_probs(h)
        post_p = model.rssm.post_probs(h, embed)
        z = model.rssm._sample(post_p)
        post_seq.append(post_p)
        prior_seq.append(prior_p)
        feat = torch.cat([h, z], dim=-1)
        if t == hist - 1:
            anchor_feat = feat
            break
        h = model.rssm.advance_h(h, z, action[:, t].to(model.torch_dtype))

    post_p = torch.stack(post_seq, dim=1)
    prior_p = torch.stack(prior_seq, dim=1)
    kl_dyn = _categorical_kl(post_p.detach(), prior_p).sum(-1).mean().clamp_min(model.free_nats)
    kl_rep = _categorical_kl(post_p, prior_p.detach()).sum(-1).mean().clamp_min(model.free_nats)
    loss_dyn = model.beta_dyn * kl_dyn
    loss_rep = model.beta_rep * kl_rep

    h_k = h
    z_k = anchor_feat[:, model.recurrent_dim :]
    prior_logits = []
    for k in range(k_horizon):
        h_k = model.rssm.advance_h(h_k, z_k, action[:, hist - 1 + k].to(model.torch_dtype))
        z_k = model.rssm._sample(model.rssm.prior_probs(h_k))
        prior_logits.append(decoder(torch.cat([h_k, z_k], dim=-1)))

    post_logits = decoder(anchor_feat)
    tgt = occ.reshape(b, k_horizon + 1, -1)
    loss_post = class_balanced_ce(post_logits, tgt[:, 0])
    prior_terms = [
        class_balanced_ce(prior_logits[k], tgt[:, k + 1]) for k in range(k_horizon)
    ]
    loss_prior = torch.stack(prior_terms).mean()
    loss_occ = loss_prior + loss_post
    loss = loss_occ + loss_dyn + loss_rep
    stats = {
        "loss": float(loss.detach().item()),
        "loss_occ": float(loss_occ.detach().item()),
        "loss_prior": float(loss_prior.detach().item()),
        "loss_post": float(loss_post.detach().item()),
        "loss_dyn": float(loss_dyn.detach().item()),
        "loss_rep": float(loss_rep.detach().item()),
    }
    return loss, stats
