#!/usr/bin/env python3
"""CUDA kernel for the Woodcock form of L_mu, forward and exact transpose.

The vectorised torch version round-trips (m,3) position and direction tensors
through global memory at every delta step, and launches a dozen kernels per
step.  Here the whole photon history lives in registers inside one launch: a
thread takes its emission voxel, samples a direction and step lengths, walks the
grid, and scores.  Only the scoring touches memory.

The estimator is the forced-interaction (expectation) one: at every delta step
score W * mu_pp/mu_max and carry W *= (1 - mu_tot/mu_max), rather than sampling
real-versus-virtual and terminating.  Unbiased -- in a homogeneous medium the
scores telescope to mu_pp/mu_tot = int mu_pp exp(-mu_tot t) dt -- and it needs
one fewer random number per step because nothing is rejected.

Two modes over the SAME paths
-----------------------------
G is self-adjoint analytically -- |r-r'| is symmetric and tau is a line integral
over the segment joining the two points -- so estimating each direction with its
own photons gives operators that agree only in expectation, to O(1/sqrt(N)).
Measured on localised probes at 1e8 photons that was a 16% disagreement between
kidney and liver, falling to 6% at 1e9: harmless if the operator is resampled
every application, fatal once it is frozen, because MLEM has no fixed point when
H^T is not the transpose of H.

So the transpose is not estimated separately.  It reuses the forward's paths and
scatters the other way.  For a path emitted at j that collides at r,

    forward    out[r] += w0[j] * W * mu_pp[r] / mu_max        (scatter)
    transpose  acc    +=         W * mu_pp[r] * y[r] / mu_max
               out[j] += w0[j] * acc                          (gather)

which is the transpose of the same realised matrix, elementwise, at any photon
count.  The gather form is also the faster of the two: one atomicAdd per photon
instead of one per step.

The path weight W starts at 1 in both modes and the emission weight is applied
at scoring time.  That matters beyond tidiness: the Russian roulette threshold
tests W, so starting W at an input-dependent weight would make the survival
depth depend on the input and the operator would not be exactly linear.

Emission is stratified, not sampled
-----------------------------------
A frozen matrix has a non-zero column only where a photon was actually emitted;
columns never sampled are exactly zero, so the forward ignores those input
voxels and the transpose, whose output lives at the emission voxel, returns zero
there.  Resampling hid this, because every application drew a different subset.

Importance sampling makes it worse, not better.  Measured with a proposal 99%
proportional to an activity reference, column coverage saturated at 20-25% however
many photons were thrown -- 20.7% at 7 groups per voxel, 24.6% at 22 -- leaving
3-17% of body voxels with an identically zero adjoint.  Coverage, not weight
variance, is the binding constraint once the operator is frozen.

So the emission voxel is not sampled at all: group g emits from voxel g mod nvox,
giving every voxel exactly K = n_groups/nvox groups.  Coverage is 100% by
construction, the proposal is exactly uniform so the transpose's emission weight
is identically 1, the counts carry no Poisson noise of their own, and the whole
question of CDFs, alias tables and presampled index arrays disappears -- along
with the memory they cost, which reached 5 GB at 1e10 photons.  Consecutive
threads take consecutive voxels, so the setup is coalesced as well.

Speed
-----
Profiled against the previous kernel at 1e8 photons on a 256x256x263 grid:

    binary search on a 17.2M-entry CDF, split mu, scatter    325 Mphot/s
    no emission search at all                                424
    + mu_tot and mu_pp interleaved as float2                 538
    + 8 directions per emission voxel                        669
    + gather instead of scatter (transpose mode)            1409

The binary search was ~24% of runtime: 24 dependent loads into 69 MB, every one
a cache miss, before the photon takes a step.  Stratification removes it outright.
Interleaving the two attenuation maps halves the number of random memory
transactions per step.  Launching several directions per emission voxel spreads
the remaining setup over more paths.

Random numbers come from a per-thread PCG32 seeded by (seed, group), so a given
seed reproduces the same photon paths regardless of launch configuration.
"""

from __future__ import annotations

import numpy as np

_SRC = r"""
extern "C" {

__device__ __forceinline__ unsigned int pcg32(unsigned long long &s) {
    unsigned long long old = s;
    s = old * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xs = (unsigned int)(((old >> 18) ^ old) >> 27);
    unsigned int rot = (unsigned int)(old >> 59);
    return (xs >> rot) | (xs << ((32 - rot) & 31));
}

__device__ __forceinline__ float urand(unsigned long long &s) {
    // 24-bit mantissa in [0,1); never returns exactly 1.
    return (float)(pcg32(s) >> 8) * (1.0f / 16777216.0f);
}

__global__ void woodcock_pp(
    const float* __restrict__ w0,       // nvox emission weight, or NULL (= 1)
    const float* __restrict__ yimg,     // nvox, sampled at collisions (TRANSPOSE)
    const float2* __restrict__ mu2,     // nvox, (mu_tot, mu_pp) interleaved
    float* __restrict__ out,            // nvox, accumulated
    const int nx, const int ny, const int nz,
    const float dx, const float dy, const float dz,
    const float mu_max, const long long n_groups, const int ndirs_total,
    const int max_steps, const unsigned long long seed)
{
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long nthr = gridDim.x * (long long)blockDim.x;
    const int nvox = nx * ny * nz;
    const int s1 = ny * nz;
    const float hx = 0.5f * nx * dx, hy = 0.5f * ny * dy, hz = 0.5f * nz * dz;
    const float cx = 0.5f * (nx - 1), cy = 0.5f * (ny - 1), cz0 = 0.5f * (nz - 1);
    const float rdx = 1.0f / dx, rdy = 1.0f / dy, rdz = 1.0f / dz;
    const float rmu = 1.0f / mu_max;

    for (long long g = tid; g < n_groups; g += nthr) {
        unsigned long long st = seed ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(g + 1));
        pcg32(st); pcg32(st);                       // decorrelate the stream

        // Stratified emission: every voxel gets exactly n_groups/nvox groups,
        // and consecutive threads take consecutive voxels.
        const int v0 = (int)(g % (long long)nvox);
        const int krep = (int)(g / (long long)nvox);   // which repeat of this voxel
        const float we = (w0 == 0) ? 1.0f : w0[v0];
#if !TRANSPOSE
        // A zero emission weight contributes nothing to the forward, and
        // skipping it changes no other photon's path: each is seeded from its
        // own group index.
        if (we == 0.0f) continue;
#endif

        int i0 = v0 / s1, rr = v0 - i0 * s1, i1 = rr / nz, i2 = rr - i1 * nz;
        const float bx = ((float)i0 - cx) * dx;
        const float by = ((float)i1 - cy) * dy;
        const float bz = ((float)i2 - cz0) * dz;

#if TRANSPOSE
        float gacc = 0.0f;
#endif
        // The ndirs_total directions this voxel gets across all its repeats are
        // laid out as a randomly rotated spherical Fibonacci set rather than
        // drawn independently: same cost, far better coverage of the sphere,
        // and still uniform marginally, so the estimator stays unbiased. The
        // rotation is seeded from the voxel alone, so every repeat of a voxel
        // shares one lattice and the strata do not collide.
        unsigned long long sv = seed ^ (0x2545F4914F6CDD1DULL * (unsigned long long)(v0 + 1));
        pcg32(sv); pcg32(sv);
        const float rot = 6.28318530718f * urand(sv);

        for (int d = 0; d < NDIR; ++d) {
            float px = bx + (urand(st) - 0.5f) * dx;
            float py = by + (urand(st) - 0.5f) * dy;
            float pz = bz + (urand(st) - 0.5f) * dz;

            int di = krep * NDIR + d;                  // 0 .. ndirs_total-1
            float uz = -1.0f + 2.0f * ((float)di + urand(st)) / (float)ndirs_total;
            float sr = sqrtf(fmaxf(0.0f, 1.0f - uz * uz));
            float sn, cs;
            __sincosf(2.39996322973f * (float)di + rot, &sn, &cs);
            float ex = sr * cs, ey = sr * sn, ez = uz;

            // W is the path weight alone, so the roulette below depends only on
            // the path and the operator stays exactly linear in its input.
            float W = 1.0f;
#if RAYMARCH
            // Deterministic quadrature of the ray integral rather than Monte
            // Carlo sampling of it. Exponential steps sample the integral with
            // M ~ mu_max*L points, so the error falls as 1/sqrt(M) while the cost
            // rises as M -- exactly break-even, which is what raising mu_max
            // alone measured. A fixed step turns the same sum into a Riemann
            // integral of a piecewise-constant integrand, so the collision-site
            // variance is not reduced but removed. The first step is jittered so
            // the sample lattice does not align systematically with the voxels.
            const float dstep = rmu;
            float frac = urand(st);
#endif
            for (int k = 0; k < max_steps; ++k) {
#if RAYMARCH
                float step = (k == 0) ? frac * dstep : dstep;
#else
                float step = -__logf(fmaxf(urand(st), 1e-20f)) * rmu;
#endif
                px += ex * step; py += ey * step; pz += ez * step;
                if (fabsf(px) >= hx || fabsf(py) >= hy || fabsf(pz) >= hz) break;

                int j0 = (int)rintf(px * rdx + cx);
                int j1 = (int)rintf(py * rdy + cy);
                int j2 = (int)rintf(pz * rdz + cz0);
                j0 = min(max(j0, 0), nx - 1);
                j1 = min(max(j1, 0), ny - 1);
                j2 = min(max(j2, 0), nz - 1);
                int vf = j0 * s1 + j1 * nz + j2;

                float2 m = mu2[vf];                 // (mu_tot, mu_pp), one load
#if RAYMARCH
                // W carries exp(-tau) exactly, so the step may be as long as the
                // voxel without the first-order error (1 - mu*ds) would incur.
                float sc = W * m.y * step;
#else
                float sc = W * m.y * rmu;
#endif
#if TRANSPOSE
                gacc += sc * yimg[vf];
#else
                atomicAdd(&out[vf], we * sc);
#endif
#if RAYMARCH
                W *= __expf(-m.x * step);
#else
                W *= (1.0f - m.x * rmu);
#endif

                // Russian roulette rather than a weight cutoff, so terminating
                // negligible histories stays unbiased.
                if (W < 1.0e-4f) {
                    if (urand(st) < 0.5f) W *= 2.0f; else break;
                }
            }
        }
#if TRANSPOSE
        atomicAdd(&out[v0], we * gacc);
#endif
    }
}

}   // extern "C"
"""

_KERNELS: dict = {}


def _kernel(ndir: int, transpose: bool, raymarch: bool):
    key = (int(ndir), bool(transpose), bool(raymarch))
    if key not in _KERNELS:
        import cupy as cp
        _KERNELS[key] = cp.RawKernel(
            _SRC, "woodcock_pp",
            options=("--use_fast_math", f"-DNDIR={int(ndir)}",
                     f"-DTRANSPOSE={1 if transpose else 0}",
                     f"-DRAYMARCH={1 if raymarch else 0}"))
    return _KERNELS[key]


def woodcock_track(w0, mu2, shape, dr, mu_max, n_groups, ndir, transpose,
                   seed, ndirs_total=None, yimg=None, max_steps=64,
                   raymarch=False,
                   threads=128, blocks=8192):
    """Run the kernel and return the accumulated score image as a torch tensor.

    All array arguments are contiguous CUDA torch tensors; the output is a torch
    tensor on the same device. cupy and torch share the allocation through
    __cuda_array_interface__, so nothing is copied to the host.
    """
    import cupy as cp
    import torch

    nx, ny, nz = (int(s) for s in shape)
    out = torch.zeros(nx * ny * nz, dtype=torch.float32, device=mu2.device)
    nul = cp.uint64(0)
    args = (cp.asarray(w0.reshape(-1)) if w0 is not None else nul,
            cp.asarray(yimg.reshape(-1)) if yimg is not None else nul,
            cp.asarray(mu2.reshape(-1)),
            cp.asarray(out),
            nx, ny, nz,
            np.float32(dr[0]), np.float32(dr[1]), np.float32(dr[2]),
            np.float32(mu_max), np.int64(n_groups),
            np.int32(ndirs_total if ndirs_total else ndir),
            np.int32(max_steps), np.uint64(seed))
    _kernel(ndir, transpose, raymarch)((blocks,), (threads,), args)
    cp.cuda.runtime.deviceSynchronize()
    return out.reshape(nx, ny, nz)
