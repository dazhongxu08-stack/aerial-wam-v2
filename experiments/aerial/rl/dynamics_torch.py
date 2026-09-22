"""Torch DreamerV3 RSSM latent world model (design doc §2.1/§2.2) — Phase 2b.

RUNS ON THE H100 ONLY (torch ``2.7.1+cu128``). ``torch`` is imported at module
top, so ``train_rl.py`` imports this module **lazily** (only when
``dynamics.kind='torch'``), keeping the stub/mock path torch-free and this whole
``rl`` package importable on the GPU-less dev host.

The loss math mirrors the pure-numpy reference in :mod:`dreamer_recipe` exactly —
symlog / two-hot / categorical-KL / free-bits / KL-balancing with the fixed loss
scales βpred=1, βdyn=1, βrep=0.1 (§1.5). That module is the single source of
truth; the torch primitives here (``_symlog``, ``_twohot_targets``,
``_categorical_kl``) are unit-tested for equality against it on tiny CPU tensors,
so the recipe is pinned without a GPU.

Follows FastWAM's module conventions (``src/fastwam/models/wan22/fastwam.py``):
``nn.Module`` subclass · explicit typed ``__init__`` kwargs · ``@classmethod
from_config(dict)`` factory · ``training_loss(sample) -> (loss, loss_dict)`` with
``forward`` delegating to it · ``save_checkpoint``/``load_checkpoint(path,
optimizer=, step=)`` · ``self.device``/``self.torch_dtype`` · ``get_logger``.
Optimizer lives in-model (AdamW betas=(0.9, 0.95) + ``clip_grad_norm_``) because
the RL corrector is a single in-loop process — deliberately NOT the
accelerate/Hydra ``Wan22Trainer`` (that orchestrates the pixel model).

§1.5 boundary: the DreamerV3 *recipe* transfers, its *tuning* does not. Do NOT
port T=16, reward-threshold 50.0, or gaze-emergence — this is a 4-D kinematic
SEARCH regime, not CTBR racing. ``MAX_IMAGINATION_HORIZON`` stays 15.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.aerial.rl.dreamer_recipe import (
    BETA_DYN,
    BETA_PRED,
    BETA_REP,
    DEFAULT_BIN_HI,
    DEFAULT_BIN_LO,
    DEFAULT_FREE_NATS,
    DEFAULT_NUM_BINS,
)
from experiments.aerial.rl.dynamics import DynamicsOutput, LatentDynamics
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.goal_features import (
    BODY_VEL_DIM,
    DEFAULT_MANEUVER_W,
    DEFAULT_REWARD_DT,
    GOAL_REL_DIM,
    REWARD_AUX_DIM,
)
from experiments.aerial.rl.wm_data import windows_to_arrays

try:  # FastWAM's distributed-safe logger when available; std logging otherwise.
    from fastwam.utils.logging_config import get_logger  # type: ignore

    logger = get_logger(__name__)
except Exception:  # pragma: no cover - fastwam not importable off-H100
    logger = logging.getLogger(__name__)

_EPS = 1e-8


# ============================================================================
# Torch mirrors of the dreamer_recipe numpy primitives (unit-tested for equality)
# ============================================================================
def _symlog(x: torch.Tensor) -> torch.Tensor:
    """``sign(x) * log(1 + |x|)`` — must match :func:`dreamer_recipe.symlog`."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _symexp(y: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_symlog` — matches :func:`dreamer_recipe.symexp`."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


def _twohot_targets(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Two-hot target distribution over ``bins`` for scalars ``x``.

    Mirrors :func:`dreamer_recipe.two_hot_encode`: mass on the two bracketing
    bins so the distribution's expectation equals ``x`` (clamped to the grid).
    Returns ``x.shape + (len(bins),)`` summing to 1 on the last axis.
    """
    k = bins.shape[0]
    xc = x.clamp(min=float(bins[0]), max=float(bins[-1]))
    # largest index with bins[lower] <= xc, clamped so upper = lower + 1 is valid.
    lower = (torch.searchsorted(bins, xc.contiguous(), right=True) - 1).clamp(0, k - 2)
    upper = lower + 1
    span = (bins[upper] - bins[lower]).clamp_min(_EPS)
    w_upper = (xc - bins[lower]) / span
    target = torch.zeros(*x.shape, k, device=x.device, dtype=x.dtype)
    target.scatter_(-1, lower.unsqueeze(-1), (1.0 - w_upper).unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), w_upper.unsqueeze(-1))
    return target


def _twohot_decode(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected bin value ``sum(probs * bins)`` — matches ``two_hot_decode``."""
    return (probs * bins).sum(dim=-1)


def _categorical_kl(post_probs: torch.Tensor, prior_probs: torch.Tensor) -> torch.Tensor:
    """KL(post ‖ prior) summed over the last (class) axis.

    Matches :func:`dreamer_recipe.categorical_kl` (same ``+_EPS`` inside the
    logs) so the torch and numpy references agree elementwise.
    """
    return (
        post_probs * (torch.log(post_probs + _EPS) - torch.log(prior_probs + _EPS))
    ).sum(dim=-1)


# ============================================================================
# Encoder / decoder / RSSM submodules
# ============================================================================
class _RGBEncoder(nn.Module):
    """Stride-2 conv stack (DreamerV3-style, depth-doubling) → flat embedding.

    RGB-only per §1.2. The flattened conv dim is measured with a dry forward so
    the module adapts to ``image_size`` without a hand-computed shape.
    """

    def __init__(self, image_size: int, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        chans = [in_ch, base, base * 2, base * 4, base * 8]
        layers = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1),
                       nn.SiLU()]
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, image_size, image_size)
            self.flat_dim = int(self.conv(dummy).flatten(1).shape[1])

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:  # rgb [N, C, H, W] in [0,1]
        return self.conv(rgb).flatten(1)


class _RGBDecoder(nn.Module):
    """Transposed-conv decoder — TRAIN-ONLY reconstruction supervision (§2.3).

    Never stepped in the online loop (the fast latent WM does not render pixels);
    it only shapes the representation and feeds the posterior-collapse detector.
    """

    def __init__(self, feature_dim: int, flat_dim: int, spatial: int,
                 out_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self._spatial = spatial
        self._c0 = base * 8
        self.fc = nn.Linear(feature_dim, self._c0 * spatial * spatial)
        chans = [base * 8, base * 4, base * 2, base, out_ch]
        layers = []
        for i, (cin, cout) in enumerate(zip(chans[:-1], chans[1:])):
            layers.append(nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1))
            if i < len(chans) - 2:
                layers.append(nn.SiLU())
        self.deconv = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:  # -> [N, C, H, W]
        x = self.fc(feature).view(-1, self._c0, self._spatial, self._spatial)
        return self.deconv(x)


class _RSSM(nn.Module):
    """Recurrent State-Space Model: deterministic ``h`` (GRU) + discrete ``z``.

    ``h`` is the real skeleton change §2.2 flagged over the stub's flat latent.
    ``z`` is a ``stoch_dim × stoch_classes`` categorical sampled straight-through
    with a small uniform mix (``unimix``) for stability (DreamerV3).
    """

    def __init__(self, embed_dim: int, action_dim: int, recurrent_dim: int,
                 stoch_dim: int, stoch_classes: int, hidden: int, unimix: float) -> None:
        super().__init__()
        self.recurrent_dim = recurrent_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.unimix = float(unimix)
        z_flat = stoch_dim * stoch_classes

        self.gru = nn.GRUCell(hidden, recurrent_dim)
        self.in_proj = nn.Sequential(nn.Linear(z_flat + action_dim, hidden), nn.SiLU())
        self.prior_net = nn.Sequential(nn.Linear(recurrent_dim, hidden), nn.SiLU(),
                                       nn.Linear(hidden, z_flat))
        self.post_net = nn.Sequential(nn.Linear(recurrent_dim + embed_dim, hidden), nn.SiLU(),
                                      nn.Linear(hidden, z_flat))

    # -- categorical helpers --------------------------------------------------
    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.view(*logits.shape[:-1], self.stoch_dim, self.stoch_classes)
        probs = F.softmax(logits, dim=-1)
        if self.unimix > 0.0:  # mix in a uniform for numerical stability (DreamerV3)
            uniform = torch.ones_like(probs) / self.stoch_classes
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
        return probs

    def _sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Straight-through one-hot sample; returns flattened ``[.., z_flat]``."""
        flat = probs.reshape(-1, self.stoch_classes)
        idx = torch.multinomial(flat, 1).squeeze(-1)
        onehot = F.one_hot(idx, self.stoch_classes).to(probs.dtype).view_as(probs)
        sample = onehot + probs - probs.detach()  # straight-through gradient
        return sample.reshape(*probs.shape[:-2], self.stoch_dim * self.stoch_classes)

    def initial_h(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.recurrent_dim, device=device, dtype=dtype)

    def prior_probs(self, h: torch.Tensor) -> torch.Tensor:
        return self._logits_to_probs(self.prior_net(h))

    def post_probs(self, h: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        return self._logits_to_probs(self.post_net(torch.cat([h, embed], dim=-1)))

    def advance_h(self, h: torch.Tensor, z_flat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.gru(self.in_proj(torch.cat([z_flat, action], dim=-1)), h)


# ============================================================================
# [1b] Multi-frame depth head (frozen §3 / §6 Step 3)
# ============================================================================
class _DepthHead(nn.Module):
    """RGB multi-frame → dense metric ``D̂`` + per-pixel ``log σ`` (frozen [1b]).

    Lives in ``dynamics_torch.py`` per frozen §9. Trained offline via
    ``train_depth_head`` on ``perception_data`` windows (GT depth never enters
    ``wm_data`` / the policy graph). ``enable`` in YAML stays false until
    ``_v0_gate`` four-signal PASS; the module itself is always constructible.
    """

    def __init__(
        self,
        *,
        image_size: int = 224,
        n_frames: int = 4,
        base: int = 32,
        motion_channels: bool = False,
        scale_factorized: bool = False,
    ) -> None:
        super().__init__()
        if int(n_frames) < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        if int(image_size) < 16 or int(image_size) % 16 != 0:
            raise ValueError(
                f"image_size must be a multiple of 16 and >= 16 (got {image_size})"
            )
        self.image_size = int(image_size)
        self.n_frames = int(n_frames)
        self.motion_channels = bool(motion_channels)
        self.scale_factorized = bool(scale_factorized)
        in_ch = self.n_frames * 3
        chs = [base, base * 2, base * 4, base * 8]
        enc: list[nn.Module] = []
        c_in = in_ch
        for c in chs:
            enc += [nn.Conv2d(c_in, c, 4, stride=2, padding=1), nn.SiLU()]
            c_in = c
        self.encoder = nn.Sequential(*enc)
        self.stem_motion: Optional[nn.Module] = None
        if self.motion_channels and self.n_frames > 1:
            # A SEPARATE stem for the frame differences, summed with the RGB stem
            # — arithmetically the same as one conv over concatenated input, but
            # it keeps the pretrained RGB stem's shape untouched and makes the
            # new pathway its own parameter tensor. That matters: Adam normalises
            # per-parameter, so a gradient-scaling hook cannot give one slice of
            # a shared weight its own step size. Only a separate tensor can go in
            # its own optimizer group, and 2026-08-06 showed why that is needed —
            # one uniform lr either leaves the new channels at their zero init
            # (lr 1e-5) or destroys the pretrained ones (lr 1e-4 broke ①d by
            # step 100).
            self.stem_motion = nn.Conv2d(
                (self.n_frames - 1) * 3, chs[0], 4, stride=2, padding=1
            )
            nn.init.zeros_(self.stem_motion.weight)
            nn.init.zeros_(self.stem_motion.bias)
        # Mirror decoder: 4× upsample back to image_size.
        dec: list[nn.Module] = []
        for c in reversed(chs[:-1]):
            dec += [nn.ConvTranspose2d(c_in, c, 4, stride=2, padding=1), nn.SiLU()]
            c_in = c
        dec += [nn.ConvTranspose2d(c_in, base, 4, stride=2, padding=1), nn.SiLU()]
        dec += [nn.Conv2d(base, 2, kernel_size=3, padding=1)]  # depth + log_sigma
        self.decoder = nn.Sequential(*dec)
        if self.scale_factorized:
            # One scalar log-scale per call, from pooled encoder features. ③ is a
            # statement about a single DOF (band depth level), so giving the Δ
            # objective one low-variance knob beats asking it to move H*W pixels
            # that AbsRel simultaneously pins down.
            self.scale_mlp = nn.Sequential(
                nn.Linear(chs[-1], chs[-1]), nn.SiLU(), nn.Linear(chs[-1], 1)
            )
            # Start at exp(0)=1 so a fresh scale-factorized net matches the plain
            # one and warm-starting from a plain ckpt is a no-op at step 0.
            nn.init.zeros_(self.scale_mlp[-1].weight)
            nn.init.zeros_(self.scale_mlp[-1].bias)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "_DepthHead":
        """Rebuild the architecture recorded in a ``train_depth_head`` checkpoint.

        Loaders must go through here: ``motion_channels`` / ``scale_factorized``
        change parameter shapes, and a loader that ignores them fails a
        ``strict=True`` load (or, worse, silently builds the wrong net).
        """
        return cls(
            image_size=int(payload.get("image_size", 224)),
            n_frames=int(payload.get("n_frames", 4)),
            base=int(payload.get("base", 32)),
            motion_channels=bool(payload.get("motion_channels", False)),
            scale_factorized=bool(payload.get("scale_factorized", False)),
        )

    @staticmethod
    def pack_rgb_nhwc(
        rgb: torch.Tensor, n_frames: int, *, motion_channels: bool = False
    ) -> torch.Tensor:
        """``rgb [B,L,H,W,3]`` uint8/float → ``[B, C, H, W]`` in [0,1].

        Uses the **last** ``n_frames`` of the window (left-pad by repeating frame 0
        when ``L < n_frames``) so a short window still yields a fixed channel count.

        ``motion_channels`` appends the ``n_frames-1`` consecutive frame
        differences, so ``C = n*3 + (n-1)*3``. Scale change over a window is a
        looming/expansion cue: it lives in the *difference* between frames, and a
        plain conv stack over concatenated RGB has to rediscover subtraction
        before it can see it. Handing it the differences is what lets the net
        estimate Δ-depth at all rather than re-guessing an absolute depth per
        call and differencing two independent guesses (2026-08-05 diagnose:
        pred-Δ vs GT-Δ Pearson ≈ 0).
        """
        if rgb.dim() != 5 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb must be [B,L,H,W,3], got {tuple(rgb.shape)}")
        B, L, H, W, _ = rgb.shape
        n = int(n_frames)
        if L >= n:
            sl = rgb[:, -n:]
        else:
            pad = rgb[:, :1].expand(B, n - L, H, W, 3)
            sl = torch.cat([pad, rgb], dim=1)
        if sl.dtype == torch.uint8:
            sl = sl.float() / 255.0
        # [B, n, H, W, 3] -> [B, n*3, H, W]
        packed = sl.permute(0, 1, 4, 2, 3).reshape(B, n * 3, H, W)
        if not motion_channels or n < 2:
            return packed
        frames = sl.permute(0, 1, 4, 2, 3)  # [B, n, 3, H, W]
        diffs = (frames[:, 1:] - frames[:, :-1]).reshape(B, (n - 1) * 3, H, W)
        return torch.cat([packed, diffs], dim=1)

    def forward(self, rgb_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``rgb_stack [B, C, H, W]`` → ``(D̂, log_σ)`` each ``[B, H, W]``.

        ``C = n_frames*3``, or ``n_frames*3 + (n_frames-1)*3`` when
        ``motion_channels`` is on (the trailing block is the frame differences,
        which go through their own stem).
        """
        if self.stem_motion is not None:
            split = self.n_frames * 3
            h = self.encoder[0](rgb_stack[:, :split])
            h = h + self.stem_motion(rgb_stack[:, split:])
            h = self.encoder[1:](h)
        else:
            h = self.encoder(rgb_stack)
        out = self.decoder(h)
        if out.shape[-2:] != (self.image_size, self.image_size):
            out = F.interpolate(
                out, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        # Softplus keeps D̂ > 0; log_σ is unconstrained.
        depth = F.softplus(out[:, 0]) + 1e-3
        log_sigma = out[:, 1]
        if self.scale_factorized:
            pooled = h.mean(dim=(-2, -1))
            # Clamp keeps a bad step from driving the whole map off the [1,40] m
            # navigational band, which would zero out ③'s support test.
            log_scale = self.scale_mlp(pooled).squeeze(-1).clamp(-2.0, 2.0)
            depth = depth * torch.exp(log_scale)[:, None, None]
        return depth, log_sigma

    def new_pathway_parameters(self) -> list:
        """Params that ``--adapt-init`` starts at zero: the Δ-scale pathway.

        These need their own optimizer group. They begin with no contribution by
        construction, so the lr that suits the pretrained weights leaves them
        parked at zero, and the lr that moves them wrecks the pretrained ones.
        """
        params: list = []
        if self.stem_motion is not None:
            params += list(self.stem_motion.parameters())
        if self.scale_factorized:
            params += list(self.scale_mlp.parameters())
        return params

    def predict_from_window(self, rgb_nhwc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: window RGB → ``(D̂, log_σ)`` for the **last** frame."""
        return self.forward(
            self.pack_rgb_nhwc(
                rgb_nhwc, self.n_frames, motion_channels=self.motion_channels
            )
        )


# ============================================================================
# [1b′] DA3 pretrained-backbone depth head (frozen §3 rev 2026-08-10)
# ============================================================================
# ImageNet stats — DA3 does NOT normalize internally (see da3 api.py preprocess).
_DA3_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_DA3_IMAGENET_STD = (0.229, 0.224, 0.225)
# Vendored pure-torch DA3 subset (see third_party/depth_anything_3/VENDOR.md,
# upstream commit 3d835ec). Weightless: fine-tuned ckpts are self-contained.
_DA3_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")


def _import_da3():
    """Lazy import of the vendored ``DinoV2`` + ``DPT`` (adds ``third_party`` to
    ``sys.path`` once). Kept lazy so importing this module never requires DA3 to
    resolve for the scratch path — only ``DA3DepthHead`` construction touches it."""
    import sys

    if _DA3_VENDOR_DIR not in sys.path:
        sys.path.insert(0, _DA3_VENDOR_DIR)
    from depth_anything_3.model.dinov2.dinov2 import DinoV2  # noqa: E402
    from depth_anything_3.model.dpt import DPT  # noqa: E402

    return DinoV2, DPT


class DA3DepthHead(nn.Module):
    """RGB single-frame → dense metric ``D̂`` via a DA3METRIC-LARGE backbone (§3′).

    A drop-in sibling of :class:`_DepthHead` (same public surface —
    ``predict_from_window`` / ``from_payload`` / ``state_dict`` / ``.encoder`` /
    ``.decoder`` / ``new_pathway_parameters``) selected by the ``backbone="da3"``
    key through :func:`build_depth_head`. ``_DepthHead`` is left byte-for-byte
    unchanged so canonical ``depth_step_5000.pt`` still rebuilds via the default
    ``"scratch"`` path.

    Wraps a DINOv2-ViT-L encoder (``.encoder``, **frozen**, warm-started from
    DA3METRIC) + a DPT depth decoder (``.decoder``, **trainable**). It bypasses
    the ``DepthAnything3Net`` wrapper (which drags omegaconf/alignment/geometry)
    and replicates only the metric-large depth forward — for metric-large
    ``alt_start=-1`` so the reference-view / cam-token branch is inert and N=1 is
    safe (VENDOR.md).

    ①d is now a pure metric depth regression (③ is solved by the reprojection
    estimator, so no Δ-loss / motion channels / scale factorization here — those
    kwargs are accepted for payload/interface parity and are inert). ``log_σ`` is
    returned as zeros; DA3 training runs ``nll_weight=0`` and runtime
    ``depth_min_pred`` uses depth only. ``224 % 14 == 0`` → no resize.
    """

    # DA3METRIC-LARGE architecture (configs/da3metric-large.yaml). Persisted in
    # the ckpt payload under ``da3_arch`` so ``from_payload`` rebuilds exactly.
    DEFAULT_ARCH: Dict[str, Any] = {
        "dino_name": "vitl",
        "out_layers": [4, 11, 17, 23],
        "dim_in": 1024,
        "dpt_features": 256,
        "dpt_out_channels": [256, 512, 1024, 1024],
    }

    def __init__(
        self,
        *,
        image_size: int = 224,
        n_frames: int = 1,
        motion_channels: bool = False,
        scale_factorized: bool = False,
        arch: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if int(image_size) < 14 or int(image_size) % 14 != 0:
            raise ValueError(
                f"DA3 requires image_size a multiple of the ViT patch size 14 "
                f"(got {image_size}); 224 = 16×14 is the aerial default."
            )
        self.image_size = int(image_size)
        # Inert for DA3 (single-frame backbone); kept so payloads / freeze logic /
        # holdout code that read these attributes behave uniformly across heads.
        self.n_frames = int(n_frames)
        self.motion_channels = bool(motion_channels)
        self.scale_factorized = bool(scale_factorized)
        self.arch = {**self.DEFAULT_ARCH, **(dict(arch) if arch else {})}

        DinoV2, DPT = _import_da3()
        self.encoder = DinoV2(
            name=self.arch["dino_name"],
            out_layers=list(self.arch["out_layers"]),
            alt_start=-1,
            qknorm_start=-1,
            rope_start=-1,
            cat_token=False,
        )
        self.decoder = DPT(
            dim_in=int(self.arch["dim_in"]),
            output_dim=1,
            features=int(self.arch["dpt_features"]),
            out_channels=list(self.arch["dpt_out_channels"]),
        )
        # ImageNet normalization applied in ``predict_from_window`` (non-persistent
        # so they never bloat / clash with a strict ckpt load).
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor(_DA3_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor(_DA3_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "DA3DepthHead":
        """Rebuild the architecture recorded in a ``--backbone da3`` checkpoint."""
        return cls(
            image_size=int(payload.get("image_size", 224)),
            n_frames=int(payload.get("n_frames", 1)),
            motion_channels=bool(payload.get("motion_channels", False)),
            scale_factorized=bool(payload.get("scale_factorized", False)),
            arch=payload.get("da3_arch", None),
        )

    def load_da3_pretrained(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, list]:
        """Warm-start encoder+decoder from a DA3METRIC-LARGE state dict.

        DA3 keys are prefixed ``model.backbone.*`` (→ our ``encoder`` = ``DinoV2``,
        whose ViT lives under ``pretrained.*``) and ``model.head.*`` (→ our
        ``decoder`` = ``DPT``). Strip the prefix, load ``strict=False`` (the sky
        head / any GS keys are dropped harmlessly). Runs on the training machine
        only; the fine-tuned ①d ckpt is self-contained thereafter.
        """
        bk, hk = {}, {}
        for k, v in state_dict.items():
            if k.startswith("model.backbone."):
                bk[k[len("model.backbone."):]] = v
            elif k.startswith("model.head."):
                hk[k[len("model.head."):]] = v
        mb = self.encoder.load_state_dict(bk, strict=False)
        mh = self.decoder.load_state_dict(hk, strict=False)
        return {
            "backbone_loaded": len(bk),
            "backbone_missing": list(mb.missing_keys),
            "backbone_unexpected": list(mb.unexpected_keys),
            "head_loaded": len(hk),
            "head_missing": list(mh.missing_keys),
            "head_unexpected": list(mh.unexpected_keys),
        }

    def _prep_last_frame(self, rgb_nhwc: torch.Tensor) -> torch.Tensor:
        """``rgb [B,L,H,W,3]`` uint8/float → ImageNet-normalized ``[B,3,H,W]``.

        Takes the window's **last** frame (matches ①d "last frame" + ③ per-frame
        reprojection). uint8 → [0,1]; float assumed already in [0,1] (mirrors
        :meth:`_DepthHead.pack_rgb_nhwc`)."""
        if rgb_nhwc.dim() != 5 or rgb_nhwc.shape[-1] != 3:
            raise ValueError(f"rgb must be [B,L,H,W,3], got {tuple(rgb_nhwc.shape)}")
        x = rgb_nhwc[:, -1].permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        return (x - self._imagenet_mean) / self._imagenet_std

    def forward(self, x_bchw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ImageNet-normalized single frame ``[B,3,H,W]`` → ``(D̂, log_σ)`` each ``[B,H,W]``.

        ``log_σ`` is a zeros tensor (DA3 trains ``nll_weight=0``)."""
        B, C, H, W = x_bchw.shape
        # DA3 backbone expects (B, N, 3, H, W); N=1 single view. alt_start=-1 →
        # cam_token / ref-view logic inert, so pass the safe defaults verbatim.
        feats, _aux = self.encoder(
            x_bchw.unsqueeze(1), cam_token=None, export_feat_layers=[]
        )
        out = self.decoder(feats, H, W, patch_start_idx=0)
        depth = out["depth"][:, 0]  # (B, S=1, H, W) → (B, H, W); exp-activated > 0
        if depth.shape[-2:] != (self.image_size, self.image_size):
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        log_sigma = torch.zeros_like(depth)
        return depth, log_sigma

    def new_pathway_parameters(self) -> list:
        """No zero-init adapter pathway — the DPT decoder trains under one lr from
        the DA3METRIC warm-start. Returned empty for interface parity with
        :meth:`_DepthHead.new_pathway_parameters`."""
        return []

    def predict_from_window(self, rgb_nhwc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: window RGB → ``(D̂, log_σ)`` for the **last** frame."""
        return self.forward(self._prep_last_frame(rgb_nhwc))


def build_depth_head(spec: Optional[Dict[str, Any]]):
    """Construct a depth head from a fresh config dict OR a checkpoint payload.

    Dispatches on the ``backbone`` key: ``"scratch"`` (default → :class:`_DepthHead`)
    or ``"da3"`` (→ :class:`DA3DepthHead`). An absent key resolves to ``"scratch"``
    so every pre-existing checkpoint — including canonical ``depth_step_5000.pt`` —
    rebuilds byte-identically. Both head classes read their arch keys via ``.get``,
    so the same factory serves fresh construction and ``from_payload`` loading.
    """
    spec = spec or {}
    backbone = str(spec.get("backbone", "scratch")).lower()
    if backbone in ("", "scratch"):
        return _DepthHead.from_payload(spec)
    if backbone == "da3":
        return DA3DepthHead.from_payload(spec)
    raise ValueError(
        f"unknown depth-head backbone {backbone!r} (expected 'scratch' or 'da3')"
    )


def depth_head_loss(
    pred: torch.Tensor,
    log_sigma: torch.Tensor,
    gt: torch.Tensor,
    *,
    absrel_weight: float = 1.0,
    silog_weight: float = 0.5,
    nll_weight: float = 0.1,
    max_depth_m: float = 200.0,
    near_weight: float = 0.0,
    near_focus_m: float = 5.0,
    near_overread_hinge_weight: float = 0.0,
    near_absrel_pinball_weight: float = 0.0,
    near_absrel_pinball_tau: float = 0.9,
    fwd_overread_hinge_weight: float = 0.0,
    near_absrel_p90_weight: float = 0.0,
    near_absrel_p90_tau: float = 0.9,
    near_fwd_absrel_pinball_weight: float = 0.0,
    near_fwd_absrel_pinball_tau: float = 0.9,
    softmin_temperature_m: float = 0.0,
    center_frac: float = 0.5,
    trigger_m: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Supervised depth loss: AbsRel L1 + mild heteroscedastic NLL on valid GT.

    AbsRel matches the V0 ①d gate metric (``v0_metrics.depth_absrel``); NLL
    trains ``σ`` without dominating early AbsRel. Invalid / non-positive GT
    pixels are masked out. Outdoor AirSim DepthPlanar is dominated by >1 km
    sky/fill — those pixels are excluded via ``max_depth_m`` so AbsRel tracks
    navigational near/mid field (safety ``depth_min_pred`` cares about the near
    end). Holdout ①d must use the same mask.

    ``near_weight`` (>0) adds a near-band emphasis term (AbsRel over GT ≤
    ``near_focus_m``). This is an ADDED training term only; it does NOT change
    the ①d gate metric or any §4.1 threshold.

    V4-⓪ declare v1 (superseded): ``near_overread_hinge_weight`` /
    signed ``near_absrel_pinball_*``.

    V4-⓪ declare v2: ``fwd_overread_hinge_weight`` + ``near_absrel_p90_weight``.

    V4-⓪ declare v3: ``silog_weight`` (P1=0), miss-aligned fwd hinge
    ``relu(D̂_fwd−trigger)`` with optional softmin on pred, hard cond on GT,
    ``near_fwd_absrel_pinball_*`` on forward crop ∩ near band.
    """
    from experiments.aerial.rl.depth_geometry import forward_min_depth_torch

    mask = (
        torch.isfinite(gt) & (gt > 1e-6) & (gt <= float(max_depth_m))
        & torch.isfinite(pred)
    )
    empty_stats = {
        "loss": 0.0,
        "absrel": float("nan"),
        "nll": 0.0,
        "silog": float("nan"),
        "n_valid": 0,
        "near_overread_hinge": float("nan"),
        "near_pinball": float("nan"),
        "fwd_overread_hinge": float("nan"),
        "fwd_hinge_hard": float("nan"),
        "near_absrel_p90": float("nan"),
        "near_fwd_absrel_pinball": float("nan"),
        "report_near_absrel_p90": float("nan"),
        "n_fwd_trigger": 0,
        "n_near_fwd": 0,
    }
    if not bool(mask.any()):
        zero = pred.sum() * 0.0
        return zero, empty_stats
    p, g, ls = pred[mask], gt[mask], log_sigma[mask]
    absrel = (torch.abs(p - g) / g).mean()
    diff = torch.log(p.clamp_min(1e-3)) - torch.log(g.clamp_min(1e-3))
    silog_var = ((diff ** 2).mean() - diff.mean() ** 2).clamp_min(0.0)
    silog = torch.sqrt(silog_var + 1e-8)
    ls_c = ls.clamp(-8.0, 8.0)
    inv_var = torch.exp(-2.0 * ls_c)
    nll = (0.5 * ((p - g) ** 2 * inv_var + 2.0 * ls_c)).mean()
    loss = (
        float(absrel_weight) * absrel
        + float(silog_weight) * silog
        + float(nll_weight) * nll
    )
    near_absrel_v = float("nan")
    near_hinge_v = float("nan")
    near_pinball_v = float("nan")
    near_p90_v = float("nan")
    report_near_p90_v = float("nan")
    near_fwd_pinball_v = float("nan")
    fwd_hinge_v = float("nan")
    fwd_hinge_hard_v = float("nan")
    n_near = 0
    n_fwd_trigger = 0
    n_near_fwd = 0
    near = g <= float(near_focus_m)
    n_near = int(near.sum().item())
    if n_near > 0:
        p_n, g_n = p[near], g[near]
        rel = (p_n - g_n) / g_n
        e_abs = torch.abs(rel)
        report_near_p90_v = float(torch.quantile(e_abs.detach(), 0.9).item())
        if float(near_weight) > 0.0:
            near_absrel = e_abs.mean()
            loss = loss + float(near_weight) * near_absrel
            near_absrel_v = float(near_absrel.detach().item())
        if float(near_overread_hinge_weight) > 0.0:
            hinge = torch.relu(rel).mean()
            loss = loss + float(near_overread_hinge_weight) * hinge
            near_hinge_v = float(hinge.detach().item())
        if float(near_absrel_pinball_weight) > 0.0:
            tau = float(near_absrel_pinball_tau)
            tau = min(max(tau, 1e-3), 1.0 - 1e-3)
            pinball = torch.where(rel >= 0, tau * rel, (1.0 - tau) * (-rel)).mean()
            loss = loss + float(near_absrel_pinball_weight) * pinball
            near_pinball_v = float(pinball.detach().item())
        if float(near_absrel_p90_weight) > 0.0:
            tau = float(near_absrel_p90_tau)
            tau = min(max(tau, 1e-3), 1.0 - 1e-3)
            med = e_abs.detach().median().clamp_min(1e-6)
            scale = (e_abs.detach() / med).clamp(1.0, 20.0)
            w = 1.0 + (2.0 * tau - 1.0) * (scale - 1.0)
            p90_term = (e_abs * w).mean()
            loss = loss + float(near_absrel_p90_weight) * p90_term
            near_p90_v = float(p90_term.detach().item())

    pred_b = pred if pred.ndim == 3 else pred.unsqueeze(0)
    gt_b = gt if gt.ndim == 3 else gt.unsqueeze(0)
    g_fwd_hard = forward_min_depth_torch(
        gt_b, center_frac=float(center_frac), softmin_temperature_m=0.0
    )
    trig = float(trigger_m)
    T_soft = float(softmin_temperature_m)

    if float(near_fwd_absrel_pinball_weight) > 0.0:
        cf = float(max(0.05, min(1.0, float(center_frac))))
        h, w = pred_b.shape[-2], pred_b.shape[-1]
        dh, dw = max(1, int(h * cf)), max(1, int(w * cf))
        r0, c0 = (h - dh) // 2, (w - dw) // 2
        pc = pred_b[:, r0 : r0 + dh, c0 : c0 + dw]
        gc = gt_b[:, r0 : r0 + dh, c0 : c0 + dw]
        band = (
            torch.isfinite(pc)
            & torch.isfinite(gc)
            & (gc > 1e-6)
            & (gc <= float(near_focus_m))
        )
        n_near_fwd = int(band.sum().item())
        if n_near_fwd == 0:
            band = (
                torch.isfinite(pc)
                & torch.isfinite(gc)
                & (gc > 1e-6)
                & (gc <= trig)
            )
            n_near_fwd = int(band.sum().item())
        if n_near_fwd > 0:
            e = torch.abs(pc[band] - gc[band]) / gc[band].clamp_min(1e-6)
            tau = float(near_fwd_absrel_pinball_tau)
            tau = min(max(tau, 1e-3), 1.0 - 1e-3)
            med = e.detach().median().clamp_min(1e-6)
            w = 1.0 + (2.0 * tau - 1.0) * ((e.detach() / med).clamp(1.0, 20.0) - 1.0)
            pin = (e * w).mean()
            loss = loss + float(near_fwd_absrel_pinball_weight) * pin
            near_fwd_pinball_v = float(pin.detach().item())

    if float(fwd_overread_hinge_weight) > 0.0:
        d_fwd_soft = forward_min_depth_torch(
            pred_b,
            center_frac=float(center_frac),
            softmin_temperature_m=T_soft,
        )
        d_fwd_hard = forward_min_depth_torch(
            pred_b, center_frac=float(center_frac), softmin_temperature_m=0.0
        )
        cond = (
            torch.isfinite(g_fwd_hard)
            & (g_fwd_hard <= trig)
            & (g_fwd_hard > 1e-6)
            & torch.isfinite(d_fwd_soft)
        )
        n_fwd_trigger = int(cond.sum().item())
        if n_fwd_trigger > 0:
            fwd_hinge = torch.relu(d_fwd_soft[cond] - trig).mean()
            loss = loss + float(fwd_overread_hinge_weight) * fwd_hinge
            fwd_hinge_v = float(fwd_hinge.detach().item())
            with torch.no_grad():
                fwd_hinge_hard_v = float(
                    torch.relu(d_fwd_hard[cond] - trig).mean().item()
                )

    return loss, {
        "loss": float(loss.detach().item()),
        "absrel": float(absrel.detach().item()),
        "silog": float(silog.detach().item()),
        "nll": float(nll.detach().item()),
        "near_absrel": near_absrel_v,
        "near_overread_hinge": near_hinge_v,
        "near_pinball": near_pinball_v,
        "near_absrel_p90": near_p90_v,
        "near_fwd_absrel_pinball": near_fwd_pinball_v,
        "report_near_absrel_p90": report_near_p90_v,
        "fwd_overread_hinge": fwd_hinge_v,
        "fwd_hinge_hard": fwd_hinge_hard_v,
        "n_near": n_near,
        "n_near_fwd": n_near_fwd,
        "n_fwd_trigger": n_fwd_trigger,
        "n_valid": int(mask.sum().item()),
    }



def _band_spatial_mean(
    depth: torch.Tensor,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    """Differentiable per-batch mean over H×W inside the navigational band → ``[B]``.

    Median is used at gate time; mean is the training surrogate (nanmedian has
    fragile / non-smooth grads that collapsed AbsRel in the 2026-08-05 delta
    retrain). Empty-band rows return NaN so the caller can mask them out.
    """
    flat = depth.reshape(depth.shape[0], -1)
    lo, hi = float(min_depth_m), float(max_depth_m)
    valid = torch.isfinite(flat) & (flat >= lo) & (flat <= hi)
    counts = valid.sum(dim=-1).clamp_min(0)
    # Zero invalid contributions; divide only where count > 0.
    masked = torch.where(valid, flat, torch.zeros_like(flat))
    sums = masked.sum(dim=-1)
    means = torch.where(
        counts > 0,
        sums / counts.clamp_min(1).to(dtype=flat.dtype),
        torch.full_like(sums, float("nan")),
    )
    return means


def depth_delta_scale_loss(
    pred_first: torch.Tensor,
    pred_last: torch.Tensor,
    gt_first: torch.Tensor,
    gt_last: torch.Tensor,
    *,
    min_depth_m: float = 1.0,
    max_depth_m: float = 40.0,
    eps: float = 1e-3,
    min_gt_delta_m: float = 0.5,
    motion_m: Optional[torch.Tensor] = None,
    support_ratio: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Temporal / Δ-depth consistency for V0 ③ (ŝ_D ≈ |Δ band-mean| in train).

    Supervises the predicted navigational-band depth change against the GT
    change (same band as frozen §4.1 ③c). Single-frame AbsRel alone does not
    teach metric scale-change — this term does. Relative form
    ``|ŝ_pred − ŝ_gt| / max(ŝ_gt, ε)`` matches the gate's relative-error spirit.

    Approach gating (2026-08-05 recipe): only rows with
    ``ŝ_gt ≥ min_gt_delta_m`` (and optionally ``ŝ_gt ≥ support_ratio · ‖Δp‖``
    when ``motion_m`` is provided) contribute. Flat / wall-parallel windows
    have near-zero GT Δ and previously dominated the batch → AbsRel collapse
    under ``delta_weight=1`` from-scratch retrains.
    """
    p0 = _band_spatial_mean(pred_first, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    p1 = _band_spatial_mean(pred_last, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    g0 = _band_spatial_mean(gt_first, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    g1 = _band_spatial_mean(gt_last, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    s_pred = torch.abs(p1 - p0)
    s_gt = torch.abs(g1 - g0)
    ok = (
        torch.isfinite(s_pred)
        & torch.isfinite(s_gt)
        & (s_gt >= max(float(eps), float(min_gt_delta_m)))
    )
    if motion_m is not None and float(support_ratio) > 0.0:
        mot = motion_m.reshape(-1).to(dtype=s_gt.dtype, device=s_gt.device)
        ok = ok & torch.isfinite(mot) & (s_gt >= float(support_ratio) * mot)
    if not bool(ok.any()):
        zero = pred_first.sum() * 0.0
        return zero, {"delta_rel": float("nan"), "n_delta": 0}
    rel = torch.abs(s_pred[ok] - s_gt[ok]) / torch.clamp(s_gt[ok], min=float(eps))
    loss = rel.mean()
    return loss, {
        "delta_rel": float(loss.detach().item()),
        "n_delta": int(ok.sum().item()),
    }


# ============================================================================
# The world model
# ============================================================================
class TorchRSSMDynamics(LatentDynamics, nn.Module):
    """DreamerV3 latent world model implementing the ``LatentDynamics`` contract.

    V1 exercises exactly two entry points: :meth:`update` (AdamW steps on replay
    windows) and :meth:`training_loss`. :meth:`encode` / :meth:`step` satisfy the
    imagination interface and run a genuine RSSM prior rollout, but online
    imagination (goal-conditioning, ``h``-threading through ``imagine()``) is a
    V4 concern — do not wire it into the corrector before that milestone.

    ``encode`` packs the latent as a single flat vector ``[h ‖ z]`` of width
    ``latent_dim = recurrent_dim + stoch_dim*stoch_classes``, so ``step(z, a)``
    carries the recurrent state without changing the numpy ``imagine()`` signature.
    """

    def __init__(
        self,
        *,
        image_size: int = 224,
        rgb_channels: int = 3,
        proprio_dim: int = 4,
        action_dim: int = 4,
        recurrent_dim: int = 512,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 256,
        num_bins: int = DEFAULT_NUM_BINS,
        bin_lo: float = DEFAULT_BIN_LO,
        bin_hi: float = DEFAULT_BIN_HI,
        free_nats: float = DEFAULT_FREE_NATS,
        beta_pred: float = BETA_PRED,
        beta_dyn: float = BETA_DYN,
        beta_rep: float = BETA_REP,
        beta_reward: float = 1.0,
        beta_depth: float = 1.0,
        unimix: float = 0.01,
        lr: float = 1e-4,
        grad_clip: float = 1000.0,
        train_steps_per_update: int = 1,
        decoder_train_only: bool = True,
        collapse_entropy_frac: float = 0.10,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
        reward_dt: float = DEFAULT_REWARD_DT,
        reward_feat_proj_dim: int = 64,
        w_maneuver: float = DEFAULT_MANEUVER_W,
    ) -> None:
        nn.Module.__init__(self)
        if int(image_size) < 16 or int(image_size) % 16 != 0:
            raise ValueError(
                f"image_size must be a multiple of 16 and >= 16 (got {image_size}): "
                "the encoder does 4 stride-2 downsamples and the decoder reconstructs "
                "via image_size//16. 224 (=14*16) is the working resolution."
            )
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.recurrent_dim = int(recurrent_dim)
        self.stoch_dim = int(stoch_dim)
        self.stoch_classes = int(stoch_classes)
        self.z_flat = self.stoch_dim * self.stoch_classes
        self.latent_dim = self.recurrent_dim + self.z_flat  # LatentDynamics attr
        self.free_nats = float(free_nats)
        self.beta_pred = float(beta_pred)
        self.beta_dyn = float(beta_dyn)
        self.beta_rep = float(beta_rep)
        self.beta_reward = float(beta_reward)
        self.beta_depth = float(beta_depth)
        self.grad_clip = float(grad_clip)
        self.train_steps_per_update = int(train_steps_per_update)
        self.decoder_train_only = bool(decoder_train_only)
        # Below this fraction of the max categorical entropy the posterior has
        # collapsed toward one-hot (§2.3); update() logs a warning if seen.
        self.collapse_entropy_frac = float(collapse_entropy_frac)

        self.encoder = _RGBEncoder(image_size, in_ch=rgb_channels)
        self.proprio_mlp = nn.Sequential(nn.Linear(self.proprio_dim, hidden_dim), nn.SiLU())
        embed_dim = self.encoder.flat_dim + hidden_dim
        self.rssm = _RSSM(embed_dim, self.action_dim, self.recurrent_dim,
                          self.stoch_dim, self.stoch_classes, hidden_dim, unimix)

        feature_dim = self.recurrent_dim + self.z_flat
        # Spatial size after the encoder's stride-2 stack (for the decoder base).
        spatial = image_size
        for _ in range(4):
            spatial //= 2
        self.decoder = _RGBDecoder(feature_dim, self.encoder.flat_dim, spatial,
                                   out_ch=rgb_channels)
        # Depth-reconstruction AUX head (2026-09-20, B'-1 gate follow-up): the
        # RGB-recon-only objective above leaves z without geometry (measured —
        # see wam_latent_depth_probe*.json, readout="weak_geometry" on every WM
        # trained before this). Never fed to the policy (spec §1.2 stays RGB+
        # proprio-only via PolicyObservation); this decoder only exists to force
        # the encoder/RSSM to make z depth-decodable during training, TRAIN-ONLY
        # like ``self.decoder`` above. Skipped automatically when a batch has no
        # GT depth (``wm_data.windows_to_arrays`` omits the key; see
        # ``training_loss``), so mixed depth/no-depth corpora train safely.
        self.depth_decoder = _RGBDecoder(feature_dim, self.encoder.flat_dim, spatial,
                                          out_ch=1)
        # Reward ≈ NavigationReward progress (Δdist to goal) needs goal + *realized*
        # motion. r60 actions are near-constant and overshoot early-horizon
        # displacement; body velocity × dt recovers it. Aux features are
        # scale-stable (unit goal dir, log dist, body vel, analytic progress).
        # Project the 1536-D RSSM feature so the 8-D aux is not drowned out.
        self.goal_rel_dim = int(GOAL_REL_DIM)
        self.body_vel_dim = int(BODY_VEL_DIM)
        self.reward_aux_dim = int(REWARD_AUX_DIM)
        self.reward_dt = float(reward_dt)
        self.w_maneuver = float(w_maneuver)
        self.reward_feat_proj_dim = int(reward_feat_proj_dim)
        self.reward_feat_proj = nn.Sequential(
            nn.Linear(feature_dim, self.reward_feat_proj_dim), nn.SiLU(),
        )
        self.reward_in_dim = (
            self.reward_feat_proj_dim + self.action_dim + self.reward_aux_dim
        )
        self.reward_head = nn.Sequential(
            nn.Linear(self.reward_in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, int(num_bins)),
        )
        self.continue_head = nn.Linear(feature_dim, 1)
        self.coll_head = nn.Linear(feature_dim, 1)
        # Directional obstacle cost: feature=[h‖z] + action → scalar ≥ 0.
        # Online reward uses this only when ``obstacle_cost_trained`` is True.
        self.obstacle_cost_head = nn.Sequential(
            nn.Linear(feature_dim + int(action_dim), hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.obstacle_cost_trained: bool = False

        self.register_buffer(
            "bins", torch.linspace(float(bin_lo), float(bin_hi), int(num_bins)))

        self.to(device=self.device, dtype=self.torch_dtype)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(lr), betas=(0.9, 0.95))
        # Optional imagination aux cache (V4): used when :meth:`step` kwargs omitted.
        self._imagine_goal_rel: Optional[np.ndarray] = None
        self._imagine_body_vel: Optional[np.ndarray] = None
        # Set True in load_checkpoint when depth_decoder.* weights actually load.
        # Random-init decoder must NOT emit d_fwd_hat (silent OA reward noise).
        self.depth_decoder_trained: bool = False

    def predict_obstacle_cost(
        self,
        feature: Any,
        action: Any,
    ) -> float:
        """Scalar obstacle cost from packed ``feature=[h‖z]`` and body action.

        Softplus so the cost is ≥ 0. Returns 0 when the head is not marked
        trained (random weights must not enter the reward). Feature width must
        equal ``latent_dim`` (= recurrent ‖ z_flat); silent z-only is rejected.
        """
        if not bool(getattr(self, "obstacle_cost_trained", False)):
            return 0.0
        feat = torch.as_tensor(feature, device=self.device, dtype=self.torch_dtype)
        act = torch.as_tensor(action, device=self.device, dtype=self.torch_dtype)
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        feat = feat.reshape(feat.shape[0], -1)
        if int(feat.shape[-1]) != int(self.latent_dim):
            raise ValueError(
                f"predict_obstacle_cost expects feature width latent_dim="
                f"{self.latent_dim} ([h‖z]), got {int(feat.shape[-1])}"
            )
        act = act.reshape(feat.shape[0], -1)[:, : int(self.action_dim)]
        if act.shape[-1] < int(self.action_dim):
            pad = torch.zeros(
                act.shape[0], int(self.action_dim) - act.shape[-1],
                device=self.device, dtype=self.torch_dtype,
            )
            act = torch.cat([act, pad], dim=-1)
        inp = torch.cat([feat, act], dim=-1)
        raw = self.obstacle_cost_head(inp).squeeze(-1)
        return float(F.softplus(raw).mean().item())

    def mark_obstacle_cost_trained(self, trained: bool = True) -> None:
        """Explicit gate for online reward / imagination (never infer from keys)."""
        self.obstacle_cost_trained = bool(trained)

    # -- factory (FastWAM from_config idiom) ---------------------------------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "TorchRSSMDynamics":
        """Build from the ``world_model:`` YAML block (see ``configs/aerial_rl.yaml``)."""
        def g(key: str, default: Any) -> Any:
            return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)

        scales = g("loss_scales", {}) or {}
        dec = g("decoder", {}) or {}
        # Prefer explicit world_model.reward_dt; else 1/env.step_hz when nested cfg.
        reward_dt = g("reward_dt", None)
        if reward_dt is None:
            reward_dt = DEFAULT_REWARD_DT
        return cls(
            image_size=int(g("image_size", 224)),
            recurrent_dim=int(g("recurrent_dim", 512)),
            stoch_dim=int(g("stoch_dim", 32)),
            stoch_classes=int(g("stoch_classes", 32)),
            num_bins=int(g("num_bins", DEFAULT_NUM_BINS)),
            bin_lo=float(g("bin_lo", DEFAULT_BIN_LO)),
            bin_hi=float(g("bin_hi", DEFAULT_BIN_HI)),
            free_nats=float(g("free_bits", DEFAULT_FREE_NATS)),
            beta_pred=float(scales.get("pred", BETA_PRED)) if isinstance(scales, dict) else BETA_PRED,
            beta_dyn=float(scales.get("dyn", BETA_DYN)) if isinstance(scales, dict) else BETA_DYN,
            beta_rep=float(scales.get("rep", BETA_REP)) if isinstance(scales, dict) else BETA_REP,
            beta_reward=float(scales.get("reward", 1.0)) if isinstance(scales, dict) else 1.0,
            beta_depth=float(scales.get("depth", 1.0)) if isinstance(scales, dict) else 1.0,
            lr=float(g("lr", 1e-4)),
            grad_clip=float(g("grad_clip", 1000.0)),
            train_steps_per_update=int(g("train_steps_per_update", 1)),
            decoder_train_only=bool(dec.get("train_only", True)) if isinstance(dec, dict) else True,
            device=str(g("device", "cuda")),
            reward_dt=float(reward_dt),
        )

    def _reward_aux_torch(
        self,
        goal_rel: torch.Tensor,
        body_vel: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Batched ``reward_aux_features`` — ``[B, L, REWARD_AUX_DIM]`` or ``[B, D]``."""
        dist = goal_rel[..., 3:4].clamp_min(1e-6)
        unit = goal_rel[..., :3] / dist
        logd = torch.log1p(dist)
        disp = body_vel * float(self.reward_dt)
        analytic = dist.squeeze(-1) - (goal_rel[..., :3] - disp).norm(dim=-1)
        analytic = analytic - float(self.w_maneuver) * action.norm(dim=-1)
        return torch.cat([unit, logd, body_vel, analytic.unsqueeze(-1)], dim=-1)

    def _reward_logits(
        self,
        feature: torch.Tensor,
        action: torch.Tensor,
        goal_rel: torch.Tensor,
        body_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Shared train/step reward head: proj(feature) ‖ a ‖ reward_aux."""
        feat_p = self.reward_feat_proj(feature)
        aux = self._reward_aux_torch(goal_rel, body_vel, action)
        return self.reward_head(torch.cat([feat_p, action, aux], dim=-1))

    # -- embedding -----------------------------------------------------------
    def _embed(self, rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """RGB (uint8 [N,H,W,3] or float [N,C,H,W]) + proprio → embedding."""
        if rgb.dim() == 4 and rgb.shape[-1] in (1, 3):     # NHWC uint8 -> NCHW [0,1]
            rgb = rgb.permute(0, 3, 1, 2).to(self.torch_dtype) / 255.0
        return torch.cat([self.encoder(rgb), self.proprio_mlp(proprio)], dim=-1)

    # -- training loss (FastWAM signature) -----------------------------------
    def training_loss(self, sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One-batch DreamerV3 world-model loss over a ``[B, L, ...]`` window.

        ``sample`` keys mirror :func:`wm_data.windows_to_arrays` (torch tensors on
        ``self.device``): ``rgb`` NHWC uint8 or normalized NCHW, ``proprio`` [B,L,4],
        ``action`` [B,L,4], ``goal_rel`` [B,L,4], ``body_vel`` [B,L,3], ``reward`` [B,L],
        ``done`` [B,L], ``collided`` [B,L]. Missing ``goal_rel`` / ``body_vel`` are
        zeros (legacy ckpt / fixtures).

        Loss = βpred·(recon + βreward·reward + continue + collision)
        + βdyn·fb(KL(sg(post)‖prior)) + βrep·fb(KL(post‖sg(prior))), with
        free-bits and KL-balancing matching :func:`dreamer_recipe.kl_balance`.
        ``βreward`` (``loss_scales.reward``, default 1) up-weights the two-hot
        reward term when pixel recon would otherwise dominate.
        """
        rgb, proprio, action = sample["rgb"], sample["proprio"], sample["action"]
        reward, done, collided = sample["reward"], sample["done"], sample["collided"]
        B, L = proprio.shape[0], proprio.shape[1]
        goal_rel = sample.get("goal_rel")
        if goal_rel is None:
            goal_rel = torch.zeros(
                B, L, self.goal_rel_dim, device=self.device, dtype=self.torch_dtype,
            )
        else:
            goal_rel = goal_rel.to(self.torch_dtype)
        body_vel = sample.get("body_vel")
        if body_vel is None:
            body_vel = torch.zeros(
                B, L, self.body_vel_dim, device=self.device, dtype=self.torch_dtype,
            )
        else:
            body_vel = body_vel.to(self.torch_dtype)

        h = self.rssm.initial_h(B, self.device, self.torch_dtype)
        z = torch.zeros(B, self.z_flat, device=self.device, dtype=self.torch_dtype)
        feats, post_probs_seq, prior_probs_seq = [], [], []
        for t in range(L):
            embed_t = self._embed(rgb[:, t], proprio[:, t])
            prior_p = self.rssm.prior_probs(h)
            post_p = self.rssm.post_probs(h, embed_t)
            z = self.rssm._sample(post_p)
            feats.append(torch.cat([h, z], dim=-1))
            post_probs_seq.append(post_p)
            prior_probs_seq.append(prior_p)
            h = self.rssm.advance_h(h, z, action[:, t].to(self.torch_dtype))

        feature = torch.stack(feats, dim=1)                 # [B, L, feature_dim]
        post_p = torch.stack(post_probs_seq, dim=1)         # [B, L, groups, classes]
        prior_p = torch.stack(prior_probs_seq, dim=1)

        # -- prediction heads (βpred) ---------------------------------------
        recon = self.decoder(feature.reshape(B * L, -1))
        rgb_target = rgb
        if rgb_target.dim() == 5 and rgb_target.shape[-1] in (1, 3):
            rgb_target = rgb_target.permute(0, 1, 4, 2, 3).to(self.torch_dtype) / 255.0
        recon_target = rgb_target.reshape(B * L, *recon.shape[1:])
        loss_recon = F.mse_loss(recon, recon_target)

        # Depth-reconstruction AUX (train-only; never in the online step/act
        # path — see __init__ comment). ``sample["depth"]`` is present only when
        # every frame in the batch carries GT depth (wm_data.windows_to_arrays);
        # symlog-compress before MSE so near (~1m) and far (~100m+) pixels don't
        # get wildly unequal gradient scale (mirrors the reward two-hot's symlog
        # target, §1.5).
        depth_sample = sample.get("depth")
        if depth_sample is not None:
            depth_gt = depth_sample.to(self.torch_dtype).reshape(B * L, *depth_sample.shape[2:])
            depth_pred = self.depth_decoder(feature.reshape(B * L, -1)).squeeze(1)
            if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                depth_pred = F.interpolate(
                    depth_pred.unsqueeze(1), size=depth_gt.shape[-2:],
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
            valid = torch.isfinite(depth_gt) & (depth_gt > 0.0) & (depth_gt < 200.0)
            if bool(valid.any()):
                target = _symlog(depth_gt)
                loss_depth_aux = F.mse_loss(depth_pred[valid], target[valid])
            else:
                loss_depth_aux = torch.zeros((), device=self.device, dtype=self.torch_dtype)
        else:
            loss_depth_aux = torch.zeros((), device=self.device, dtype=self.torch_dtype)

        # Goal+velocity-conditioned reward: same concat timing as :meth:`step`.
        act = action.to(self.torch_dtype)
        reward_logits = self._reward_logits(feature, act, goal_rel, body_vel)
        reward_target = _twohot_targets(_symlog(reward.to(self.torch_dtype)), self.bins)
        loss_reward = -(reward_target * F.log_softmax(reward_logits, dim=-1)).sum(-1).mean()

        cont_target = (~done.bool()).to(self.torch_dtype)   # continue = 1 - done
        loss_cont = F.binary_cross_entropy_with_logits(
            self.continue_head(feature).squeeze(-1), cont_target)
        loss_coll = F.binary_cross_entropy_with_logits(
            self.coll_head(feature).squeeze(-1), collided.to(self.torch_dtype))

        # ``beta_reward`` up-weights the two-hot reward term inside βpred (recon
        # MSE on 224² frames otherwise dwarfs reward CE; V1-② fidelity needs it).
        loss_pred = (
            loss_recon + self.beta_reward * loss_reward + loss_cont + loss_coll
            + self.beta_depth * loss_depth_aux
        )

        # -- dynamics / representation KL (βdyn / βrep) ---------------------
        kl_dyn = _categorical_kl(post_p.detach(), prior_p).sum(-1)   # KL(sg(post)‖prior)
        kl_rep = _categorical_kl(post_p, prior_p.detach()).sum(-1)   # KL(post‖sg(prior))
        kl_dyn = kl_dyn.mean().clamp_min(self.free_nats)             # free bits
        kl_rep = kl_rep.mean().clamp_min(self.free_nats)
        loss_dyn = self.beta_dyn * kl_dyn
        loss_rep = self.beta_rep * kl_rep

        loss_total = self.beta_pred * loss_pred + loss_dyn + loss_rep

        # posterior-collapse telemetry (§2.3): mean categorical entropy vs. max.
        with torch.no_grad():
            ent = -(post_p * torch.log(post_p + _EPS)).sum(-1).mean()
            max_ent = float(np.log(self.stoch_classes))
        loss_dict = {
            "loss": float(loss_total.detach().item()),
            "loss_pred": float((self.beta_pred * loss_pred).detach().item()),
            "loss_dyn": float(loss_dyn.detach().item()),
            "loss_rep": float(loss_rep.detach().item()),
            "loss_recon": float(loss_recon.detach().item()),
            "loss_reward": float(loss_reward.detach().item()),
            "loss_depth_aux": float(loss_depth_aux.detach().item()),
            "has_depth_batch": bool(depth_sample is not None),
            "recon_err": float(loss_recon.detach().item()),
            "post_entropy": float(ent.item()),
            "post_entropy_frac": float(ent.item() / max_ent) if max_ent > 0 else 0.0,
        }
        return loss_total, loss_dict

    def forward(self, *args, **kwargs):  # FastWAM aliases forward -> training_loss
        return self.training_loss(*args, **kwargs)

    def apply_freeze_backbone_train_reward_head(self) -> list:
        """Freeze encoder/RSSM/decoder/cont/coll; train ``reward_feat_proj`` + head only."""
        for param in self.parameters():
            param.requires_grad = False
        head_params = list(self.reward_feat_proj.parameters()) + list(
            self.reward_head.parameters()
        )
        for param in head_params:
            param.requires_grad = True
        return [p for p in head_params if p.requires_grad]

    def _sequence_features(
        self,
        sample: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """RSSM teacher-forcing features ``[B,L,D]`` + reward labels/conditioning."""
        rgb, proprio, action = sample["rgb"], sample["proprio"], sample["action"]
        reward, done, collided = sample["reward"], sample["done"], sample["collided"]
        del done, collided  # reward-head finetune does not use cont/coll
        B, L = proprio.shape[0], proprio.shape[1]
        goal_rel = sample.get("goal_rel")
        if goal_rel is None:
            goal_rel = torch.zeros(
                B, L, self.goal_rel_dim, device=self.device, dtype=self.torch_dtype,
            )
        else:
            goal_rel = goal_rel.to(self.torch_dtype)
        body_vel = sample.get("body_vel")
        if body_vel is None:
            body_vel = torch.zeros(
                B, L, self.body_vel_dim, device=self.device, dtype=self.torch_dtype,
            )
        else:
            body_vel = body_vel.to(self.torch_dtype)

        h = self.rssm.initial_h(B, self.device, self.torch_dtype)
        z = torch.zeros(B, self.z_flat, device=self.device, dtype=self.torch_dtype)
        feats = []
        for t in range(L):
            embed_t = self._embed(rgb[:, t], proprio[:, t])
            post_p = self.rssm.post_probs(h, embed_t)
            z = self.rssm._sample(post_p)
            feats.append(torch.cat([h, z], dim=-1))
            h = self.rssm.advance_h(h, z, action[:, t].to(self.torch_dtype))
        feature = torch.stack(feats, dim=1)
        return feature, action.to(self.torch_dtype), goal_rel, body_vel, reward

    def training_loss_reward_head(
        self, sample: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Two-hot reward CE only; backbone forward under ``no_grad``."""
        with torch.no_grad():
            feature, act, goal_rel, body_vel, reward = self._sequence_features(sample)
        reward_logits = self._reward_logits(feature, act, goal_rel, body_vel)
        reward_target = _twohot_targets(_symlog(reward.to(self.torch_dtype)), self.bins)
        loss_reward = -(reward_target * F.log_softmax(reward_logits, dim=-1)).sum(-1).mean()
        return loss_reward, {
            "loss": float(loss_reward.detach().item()),
            "loss_reward": float(loss_reward.detach().item()),
        }

    def update_reward_head(
        self,
        windows: Any,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """One AdamW step on ``reward_feat_proj`` + ``reward_head`` only."""
        arrays = windows_to_arrays(windows)
        sample = self._arrays_to_tensors(arrays)
        self.train()
        opt = optimizer if optimizer is not None else self.optimizer
        loss, last = self.training_loss_reward_head(sample)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(
            [p for p in self.parameters() if p.requires_grad], self.grad_clip,
        ))
        opt.step()
        return {"grad_norm": grad_norm, **last}

    def set_imagination_aux(
        self,
        goal_rel: Optional[np.ndarray] = None,
        body_vel: Optional[np.ndarray] = None,
    ) -> None:
        """Cache aux for :meth:`step` when ``goal_rel`` / ``body_vel`` kwargs omitted."""
        if goal_rel is None:
            self._imagine_goal_rel = None
        else:
            self._imagine_goal_rel = np.asarray(goal_rel, dtype=np.float32).reshape(
                self.goal_rel_dim,
            )
        if body_vel is None:
            self._imagine_body_vel = None
        else:
            self._imagine_body_vel = np.asarray(body_vel, dtype=np.float32).reshape(
                self.body_vel_dim,
            )

    # -- V1 gate: real world-model update ------------------------------------
    def update(self, windows: Any) -> Dict[str, Any]:
        """Run ``train_steps_per_update`` AdamW steps on replay ``windows``.

        Returns the last step's loss breakdown with NO ``skipped`` key, so
        ``corrector._update_world_model`` reports status ``"updated"`` (a real
        training step happened). Raises nothing on well-formed windows; the
        corrector guards empty/insufficient buffers before calling.
        """
        arrays = windows_to_arrays(windows)
        sample = self._arrays_to_tensors(arrays)
        self.train()
        last: Dict[str, float] = {}
        grad_norm = 0.0
        for _ in range(self.train_steps_per_update):
            loss, last = self.training_loss(sample)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip))
            self.optimizer.step()
        if last.get("post_entropy_frac", 1.0) < self.collapse_entropy_frac:
            logger.warning(
                "posterior-collapse watch: entropy at %.1f%% of max (< %.0f%%); "
                "the discrete latent is near one-hot (§2.3)",
                100.0 * last.get("post_entropy_frac", 0.0),
                100.0 * self.collapse_entropy_frac,
            )
        return {"grad_norm": grad_norm, **last}

    def _arrays_to_tensors(self, arrays: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in arrays.items():
            t = torch.from_numpy(np.ascontiguousarray(v))
            if k in ("proprio", "action", "reward", "depth", "goal_rel", "body_vel"):
                t = t.to(self.torch_dtype)
            out[k] = t.to(self.device)
        return out

    # -- imagination interface (V4; see class docstring) --------------------
    @torch.no_grad()
    def encode(self, obs: Observation) -> np.ndarray:
        """Single observation → packed latent ``[h ‖ z]`` (numpy, width latent_dim).

        No history is available for a lone observation, so ``h`` starts at zero and
        ``z`` is the posterior sample given that ``h`` — the standard RSSM initial
        state. Packing ``h`` inside the returned vector lets :meth:`step` advance
        the recurrent state without changing the numpy ``imagine()`` signature.
        """
        self.eval()
        rgb = torch.from_numpy(np.ascontiguousarray(obs.rgb)).unsqueeze(0).to(self.device)
        proprio = torch.from_numpy(
            np.ascontiguousarray(obs.proprio4())).to(self.torch_dtype).unsqueeze(0).to(self.device)
        h = self.rssm.initial_h(1, self.device, self.torch_dtype)
        embed = self._embed(rgb, proprio)
        z = self.rssm._sample(self.rssm.post_probs(h, embed))
        return torch.cat([h, z], dim=-1).squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def observe_and_advance(
        self,
        prev_latent: np.ndarray,
        action: np.ndarray,
        current_obs: Observation,
    ) -> np.ndarray:
        """Streamed posterior: advance ``h`` with action, update ``z`` from obs."""
        self.eval()
        prev = torch.from_numpy(np.ascontiguousarray(prev_latent)).to(
            self.device, self.torch_dtype
        ).reshape(1, self.latent_dim)
        h_prev = prev[:, : self.recurrent_dim]
        z_prev = prev[:, self.recurrent_dim :]
        a = torch.from_numpy(np.ascontiguousarray(action)).to(
            self.device, self.torch_dtype
        ).reshape(1, self.action_dim)
        h_next = self.rssm.advance_h(h_prev, z_prev, a)

        rgb = torch.from_numpy(np.ascontiguousarray(current_obs.rgb)).unsqueeze(0).to(self.device)
        proprio = torch.from_numpy(
            np.ascontiguousarray(current_obs.proprio4())
        ).to(self.torch_dtype).unsqueeze(0).to(self.device)
        embed = self._embed(rgb, proprio)
        z_next = self.rssm._sample(self.rssm.post_probs(h_next, embed))
        return torch.cat([h_next, z_next], dim=-1).squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def step(
        self,
        z: np.ndarray,
        action: np.ndarray,
        goal_rel: Optional[np.ndarray] = None,
        body_vel: Optional[np.ndarray] = None,
    ) -> DynamicsOutput:
        """One imagined RSSM prior transition from packed latent ``[h ‖ z]``.

        Head alignment (V1-②): ``training_loss`` scores cont/coll from the
        *pre-action* feature ``[h_t ‖ z_t]`` and reward from
        ``proj([h_t ‖ z_t]) ‖ a_t ‖ reward_aux(goal_rel, body_vel, a)`` against
        ``reward_t`` / ``done_t`` / ``collided_t``. Imagination / open-loop
        fidelity must match — read heads on the current feature, *then* advance.

        ``goal_rel`` and ``body_vel`` are teacher-forced from the recorded pose
        during fidelity eval (same protocol as recorded actions). When omitted,
        zeros are used (Stub / imagination paths without a goal/velocity).

        V4 note: ``progress`` is the reward head's symexp readout (per-step
        reward); goal-conditioned progress / arrival finalize at V4.
        """
        self.eval()
        packed = torch.from_numpy(np.ascontiguousarray(z)).to(
            self.device, self.torch_dtype).reshape(1, self.latent_dim)
        h = packed[:, : self.recurrent_dim]
        z_flat = packed[:, self.recurrent_dim :]
        a = torch.from_numpy(np.ascontiguousarray(action)).to(
            self.device, self.torch_dtype).reshape(1, self.action_dim)
        if goal_rel is None:
            if self._imagine_goal_rel is not None:
                g = torch.from_numpy(np.ascontiguousarray(self._imagine_goal_rel)).to(
                    self.device, self.torch_dtype,
                ).reshape(1, self.goal_rel_dim)
            else:
                g = torch.zeros(1, self.goal_rel_dim, device=self.device, dtype=self.torch_dtype)
        else:
            g = torch.from_numpy(np.ascontiguousarray(goal_rel)).to(
                self.device, self.torch_dtype).reshape(1, self.goal_rel_dim)
        if body_vel is None:
            if self._imagine_body_vel is not None:
                v = torch.from_numpy(np.ascontiguousarray(self._imagine_body_vel)).to(
                    self.device, self.torch_dtype,
                ).reshape(1, self.body_vel_dim)
            else:
                v = torch.zeros(1, self.body_vel_dim, device=self.device, dtype=self.torch_dtype)
        else:
            v = torch.from_numpy(np.ascontiguousarray(body_vel)).to(
                self.device, self.torch_dtype).reshape(1, self.body_vel_dim)

        feature = torch.cat([h, z_flat], dim=-1)
        reward_probs = F.softmax(self._reward_logits(feature, a, g, v), dim=-1)
        progress = float(_symexp(_twohot_decode(reward_probs, self.bins)).item())
        p_coll = float(torch.sigmoid(self.coll_head(feature)).item())
        cont = float(torch.sigmoid(self.continue_head(feature)).item())
        done = bool(cont < 0.5 or p_coll >= 1.0)

        # Depth-aux clearance readout (train-time decoder reused at imagination
        # time — never fed to the policy). Centre-patch median of the symexp'd
        # depth map ≈ forward clearance. Only emit when checkpoint actually
        # loaded depth_decoder.* — pre-aux / random-init weights would inject
        # garbage clearance_risk into OA rewards (2026-09-21 audit).
        d_fwd_hat: Optional[float] = None
        depth_dec = getattr(self, "depth_decoder", None)
        if depth_dec is not None and bool(getattr(self, "depth_decoder_trained", False)):
            depth_sym = depth_dec(feature).squeeze(0).squeeze(0)  # [H, W] or [1,H,W]
            if depth_sym.ndim == 3:
                depth_sym = depth_sym[0]
            depth_m = _symexp(depth_sym)
            hh, ww = int(depth_m.shape[-2]), int(depth_m.shape[-1])
            y0, y1 = hh // 3, 2 * hh // 3
            x0, x1 = ww // 3, 2 * ww // 3
            patch = depth_m[y0:y1, x0:x1].reshape(-1)
            finite = patch[torch.isfinite(patch) & (patch > 0)]
            if finite.numel() > 0:
                d_fwd_hat = float(finite.median().item())

        obstacle_cost: Optional[float] = None
        if bool(getattr(self, "obstacle_cost_trained", False)):
            inp = torch.cat([feature, a], dim=-1)
            obstacle_cost = float(F.softplus(self.obstacle_cost_head(inp)).reshape(-1)[0].item())

        h_next = self.rssm.advance_h(h, z_flat, a)
        z_next = self.rssm._sample(self.rssm.prior_probs(h_next))
        packed_next = torch.cat([h_next, z_next], dim=-1).squeeze(0).float().cpu().numpy()
        return DynamicsOutput(
            z_next=packed_next,
            p_coll=p_coll,
            progress=progress,
            done=done,
            arrived=False,
            d_fwd_hat=d_fwd_hat,
            obstacle_cost=obstacle_cost,
        )

    # -- checkpoint I/O (FastWAM idiom: path + optional optimizer/step) ------
    def save_checkpoint(self, path: str, optimizer: Optional[Any] = None,
                        step: Optional[int] = None) -> None:
        payload: Dict[str, Any] = {
            "model": self.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            # Explicit flags — presence of head keys ≠ trained (architecture
            # always includes obstacle_cost_head after 2026-09-21).
            "obstacle_cost_trained": bool(
                getattr(self, "obstacle_cost_trained", False)
            ),
            "depth_decoder_trained": bool(
                getattr(self, "depth_decoder_trained", False)
            ),
        }
        opt = optimizer if optimizer is not None else self.optimizer
        if opt is not None:
            payload["optimizer"] = opt.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str, optimizer: Optional[Any] = None) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu")
        if "model" in payload:
            filtered, skipped = _filter_compatible_state_dict(self, payload["model"])
            missing, unexpected = self.load_state_dict(filtered, strict=False)
            payload["load_skipped"] = skipped
            payload["load_missing"] = list(missing)
            payload["load_unexpected"] = list(unexpected)
            n_depth = sum(1 for k in filtered if k.startswith("depth_decoder."))
            # Depth decoder still uses key-presence (legacy aux WMs omit the
            # module entirely when untrained). Prefer explicit flag when set.
            if "depth_decoder_trained" in payload:
                self.depth_decoder_trained = bool(payload["depth_decoder_trained"])
            else:
                self.depth_decoder_trained = n_depth > 0
            payload["depth_decoder_trained"] = bool(self.depth_decoder_trained)
            payload["depth_decoder_keys_loaded"] = int(n_depth)
            n_obs = sum(1 for k in filtered if k.startswith("obstacle_cost_head."))
            payload["obstacle_cost_keys_loaded"] = int(n_obs)
            # NEVER infer True from key presence alone — random-init heads are
            # always in state_dict after architecture land.
            if "obstacle_cost_trained" in payload:
                self.obstacle_cost_trained = bool(payload["obstacle_cost_trained"])
            else:
                self.obstacle_cost_trained = False
            if self.obstacle_cost_trained and n_obs == 0:
                self.obstacle_cost_trained = False
            payload["obstacle_cost_trained"] = bool(self.obstacle_cost_trained)
        opt = optimizer if optimizer is not None else self.optimizer
        if opt is not None and "optimizer" in payload:
            try:
                opt.load_state_dict(payload["optimizer"])
            except (ValueError, RuntimeError) as exc:
                payload["optimizer_load_skipped"] = str(exc)
        return payload


def _filter_compatible_state_dict(
    model: nn.Module, state: Dict[str, Any]
) -> tuple[Dict[str, Any], list[str]]:
    """Keep only checkpoint tensors whose shapes match the live model.

    PyTorch ``strict=False`` still raises on size mismatch; legacy WM ckpts saved
    before the V1-② reward-head refactor need encoder/RSSM weights without
    forcing a full architecture match.
    """
    tgt = model.state_dict()
    filtered: Dict[str, Any] = {}
    skipped: list[str] = []
    for key, val in state.items():
        if key not in tgt:
            continue
        if tuple(tgt[key].shape) != tuple(val.shape):
            skipped.append(f"{key}: ckpt{tuple(val.shape)} vs model{tuple(tgt[key].shape)}")
            continue
        filtered[key] = val
    return filtered, skipped
