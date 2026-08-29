#!/usr/bin/env python3
"""L_mu by Woodcock tracking at native voxel resolution, on a CUDA kernel.

Why this replaces the ray-marching form in pp_vertex_het.py.  That operator
computes Phi on a coarse isotropic grid and interpolates back, G_hat = Pi G_c R,
whose true transpose is R^T G_c^T Pi^T.  Trilinear restriction is not the adjoint
of trilinear prolongation, so applying G_hat a second time is not the adjoint.
Measured by dot-product test: G_c alone is symmetric to 0.001%, G_hat to 0.03%,
but weighted by diag(mu_pp) -- which spans 3209x across the body and
concentrates in thin bone, exactly the high-frequency detail restriction throws
away -- the error reaches 1.26%.  Symmetrising the resampling pair does not help
(the mu_pp-weighted error went 1.30% -> 1.62%), because any coarse grid loses the
structure mu_pp selects for.

Woodcock tracking has no coarse grid, no restriction and no prolongation, so the
cause is removed rather than corrected.  The only discretisation left is the
voxel grid the attenuation maps already live on.

Estimator.  The density of first real interactions at r for a source f is
mu_tot(r) Phi(r), so scoring weight w at each collision estimates
integral w mu_tot Phi:

    forward    L_mu f   = b mu_pp Phi[f]       source f,          w = mu_pp/mu_tot
    adjoint    L_mu^T g = b Phi[mu_pp g]       source mu_pp * g,  w = 1/mu_tot

The interaction is *forced* rather than sampled: at every delta step the kernel
scores W * (mu_tot/mu_max) * w and carries W *= (1 - mu_tot/mu_max).  Unbiased --
in a homogeneous medium the scores telescope to mu_pp/mu_tot, which is exactly
int mu_pp exp(-mu_tot t) dt -- and better than accepting or rejecting on both
counts: no photon is discarded, and it needs one fewer random number per step.

Two things had to be true for this to work inside MLEM, and neither was at first.

The operator has to be the same matrix at every application.  Drawing the
emission voxel from the input's own distribution reads as the obvious choice and
is wrong: the random stream can be frozen, but a changed input sends the same
numbers to different voxels, so the paths move anyway.  Measured, perturbing the
input by 5% -- less than one MLEM update -- moved the output as much as a
completely fresh seed (sd log 0.243 against 0.241).  Because the update is
multiplicative the estimate then picks up an independent factor each
subiteration, a random walk in log space growing as sqrt(n) without bound:
measured exponent 0.491 against 0.500, reaching sd log 4.8 by iteration 100 at
1e8 photons.  No photon count fixes that, since suppressing sqrt(n) growth by
brute force costs n photons.  Sampling instead from a fixed proposal q, with each
photon carrying w = x_j/q_j, makes the paths identical at every application and
the operator exactly linear rather than linear in expectation.

The adjoint has to be the transpose of the realised matrix, not an independent
estimate of the same limit.  G is self-adjoint analytically, so in the limit of
infinite photons the two coincide; at finite N they differ by O(1/sqrt(N)), which
measured 16% between kidney and liver at 1e8 photons and 6% at 1e9.  Resampling
averages that away, freezing does not, and MLEM has no fixed point when H^T is
not the transpose of H.  So the transpose reuses the forward's own paths and
scatters the other way -- see pp_vertex_mc_cuda -- which is exact at any N and
happens to be the faster direction as well.

q is uniform, and stratified rather than sampled: group g emits from voxel
g mod nvox, so every voxel gets exactly n_groups/nvox groups.  An importance
proposal built from an activity reference is the obvious refinement and is a
mistake here.  A frozen matrix has a non-zero column only where a photon was
emitted, and concentrating the proposal starves the rest: measured with 99% of
the proposal following such a reference, column coverage saturated at 20-25%
however many photons were thrown, leaving 3-17% of body voxels with an
identically zero adjoint.  Coverage is the binding constraint once the operator
is frozen, and uniform stratification gives 100% of it for free.

What that leaves is a single number to choose: K = n_groups/nvox, the groups per
voxel, which sets the variance and nothing else.
"""

from __future__ import annotations

import numpy as np
import torch

from pp_vertex_mc_cuda import woodcock_track

try:
    import pytomography
    from pytomography.transforms import Transform
    _HAVE_PYTOMOGRAPHY = True
except Exception:                                    # standalone use
    _HAVE_PYTOMOGRAPHY = False

    class Transform:                                 # minimal stand-in
        def configure(self, object_meta, proj_meta=None):
            self.object_meta = object_meta


class MCPPVertexTransform(Transform):
    """L_mu = b diag(mu_pp) G, with G tracked by a CUDA Woodcock kernel.

    Args:
        mu_tot: total linear attenuation at the high-energy line (1/mm).
        mu_pp:  pair-production linear attenuation (1/mm), same shape.
        b:      branching ratio; a pure scale.
        n_photons: photons per application.  Sets the variance and nothing else;
            the estimator is unbiased at any count.  Rounded down to a whole
            number of groups per voxel, so the stratification stays exact.
        ndir: directions launched per emission voxel.  Amortises the position
            setup over several paths; 8 was worth about 25% throughput in
            profiling.
        max_steps: cap on delta steps.  The mean step is 1/mu_max ~ 93 mm
            against an 86 cm grid, so crossing the whole field takes ~10 steps
            and 64 leaves room for the tail.  Decayed histories are ended by
            Russian roulette inside the kernel rather than by a weight cutoff,
            so this cap is the only approximation and it sits far out in the tail.
        frozen: reuse the same seed at every application.  With stratified
            emission this makes the operator a genuine fixed matrix; there is no
            reason to turn it off except to reproduce the old behaviour.

    The operator is a fixed matrix as soon as setup() has run.  Nothing about it
    depends on the input except linearly, and its adjoint is the transpose of the
    matrix it actually realised rather than a second estimate of the same limit.
    """

    def __init__(self, mu_tot, mu_pp, b: float = 1.0,
                 n_photons: int = 1_000_000_000, ndir: int = 8,
                 max_steps: int | None = None, mu_max_scale: float = 1.0,
                 raymarch: bool = False,
                 n_photons_sens: int | None = None,
                 seed: int = 0, frozen: bool = True,
                 threads: int = 128, blocks: int = 8192):
        super().__init__()
        self.mu_tot_fine = torch.as_tensor(np.asarray(mu_tot), dtype=torch.float32)
        self.mu_pp_fine = torch.as_tensor(np.asarray(mu_pp), dtype=torch.float32)
        if self.mu_tot_fine.shape != self.mu_pp_fine.shape:
            raise ValueError("mu_tot and mu_pp must have the same shape")
        self.b = float(b)
        self.ndir = max(1, int(ndir))
        self.n_photons = int(n_photons)
        # Once the proposal is fixed, numerator and denominator share the same
        # matrix and their errors cancel in the ratio, so the sensitivity image
        # no longer needs a quieter pass of its own.
        self.n_photons_sens = int(n_photons_sens if n_photons_sens is not None
                                  else n_photons)
        # Raising mu_max shortens the delta step, so the photon scores more often
        # along the same ray with proportionally smaller weights. As mu_max grows
        # the sum converges to the deterministic ray integral int mu_pp exp(-tau)
        # dt and the collision-site randomness disappears, leaving only direction
        # and emission-position variance. The step budget has to grow with it.
        self.mu_max_scale = float(mu_max_scale)
        self.raymarch = bool(raymarch)
        self.max_steps = int(max_steps if max_steps is not None
                             else round((10 if self.raymarch else 64)
                                        * self.mu_max_scale))
        self.seed = int(seed)
        self.frozen = bool(frozen)
        self.threads, self.blocks = int(threads), int(blocks)
        self._epoch = 0

    # ------------------------------------------------------------------
    def configure(self, object_meta, proj_meta=None) -> None:
        super().configure(object_meta, proj_meta)
        self.setup(tuple(int(n) for n in object_meta.shape),
                   tuple(float(d) for d in object_meta.dr))

    def setup(self, shape, dr, device=None) -> None:
        self.device = device or (
            pytomography.device if _HAVE_PYTOMOGRAPHY
            else ("cuda" if torch.cuda.is_available() else "cpu"))
        if not str(self.device).startswith("cuda"):
            raise RuntimeError(
                "MCPPVertexTransform runs its tracker as a CUDA kernel and has "
                "no CPU path; a GPU is required.")
        if tuple(self.mu_tot_fine.shape) != tuple(shape):
            raise ValueError(f"mu shape {tuple(self.mu_tot_fine.shape)} != "
                             f"object shape {tuple(shape)}")
        self.shape = tuple(shape)
        self.nvox = int(np.prod(self.shape))
        self.dr = tuple(float(d) for d in dr)
        self.mu_tot = self.mu_tot_fine.to(self.device).contiguous()
        self.mu_pp = self.mu_pp_fine.to(self.device).contiguous()
        # One interleaved (mu_tot, mu_pp) array: the kernel reads both at the
        # same random index every step, so one 8-byte load replaces two.
        self.mu2 = torch.stack([self.mu_tot.reshape(-1),
                                self.mu_pp.reshape(-1)], 1).reshape(-1).contiguous()
        # A margin keeps mu_tot/mu_max strictly below 1 even if a resampled map
        # produces a value fractionally above the stored maximum.
        self.mu_max = float(self.mu_tot.max()) * 1.0001 * self.mu_max_scale

    # ------------------------------------------------------------------
    @property
    def K(self) -> int:
        """Groups per voxel. The only quantity that sets the variance."""
        return max(1, self.n_photons // (self.ndir * self.nvox))

    @property
    def n_groups(self) -> int:
        return self.K * self.nvox

    # ------------------------------------------------------------------
    def _run(self, w0, transpose, yimg=None):
        if not self.frozen:
            self._epoch += 1
        seed = self.seed if self.frozen else self.seed + 7919 * self._epoch
        return woodcock_track(
            w0, self.mu2, self.shape, self.dr, self.mu_max,
            self.n_groups, self.ndir, transpose, seed,
            ndirs_total=self.K * self.ndir, yimg=yimg,
            raymarch=self.raymarch,
            max_steps=self.max_steps, threads=self.threads, blocks=self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """b * mu_pp * Phi[x]."""
        x = x.to(self.device)
        flat = x.reshape(-1)
        tot = float(flat.sum())
        if tot == 0:
            return torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        # w = x_j / q_j with q = 1/nvox, carried in units of the mean so the
        # weights are O(1) and the atomicAdd accumulation keeps its precision.
        w0 = (flat * (self.nvox / tot)).contiguous()
        out = self._run(w0, False)
        return self.b * out * (tot / (self.n_groups * self.ndir))

    def backward(self, x: torch.Tensor, norm_constant=None):
        """The exact transpose of what forward() realised.

        With a uniform proposal the emission weight is identically 1, so the
        kernel needs no weight array at all in this direction.
        """
        x = x.to(self.device)
        flat = x.reshape(-1)
        m = float(flat.abs().mean())
        if m == 0:
            out = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        else:
            out = self._run(None, True, yimg=(flat / m).contiguous())
            out = self.b * out * (m * self.nvox / (self.n_groups * self.ndir))
        return (out, norm_constant) if norm_constant is not None else out
