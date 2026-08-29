# Integrating the pair-production transport operator

This file is written for both a coding agent and a human. It gives the context
needed to place `MCPPVertexTransform` inside a PyTomography reconstruction
without breaking the properties the operator depends on.

## What the operator is

`MCPPVertexTransform` implements

```
  Gamma f(r) = b * mu_pp(r) * INT exp(-tau(r,r')) / (4 pi |r-r'|^2) f(r') d3r'
```

mapping an activity density to the density of pair-production vertices. `tau`
is the line integral of `mu_tot` between the two points. It answers the
question "given activity here, where do the high-energy photons convert", and
the conversion sites are what a PET scanner actually records.

For 212Pb the photon is the 2.6145 MeV emission of 208Tl, two decays
downstream, with a mean free path of about 24 cm in soft tissue. The
displacement between decay and conversion is comparable to the patient, so a
reconstruction that omits this operator converges to the vertex density rather
than to the activity.

The integral is evaluated by Woodcock delta tracking in a CUDA kernel
(`pp_vertex_mc_cuda.py`). There is no CPU path.

## Where it goes in the system matrix

```
  H = N A P B Gamma
```

- `N`  normalisation: detector efficiency, geometric sensitivity
- `A`  511 keV attenuation factors
- `P`  geometric projector (scanner geometry)
- `B`  image-space resolution model
- `Gamma`  this operator

Read right to left, the factors are in the order the physics happens: the decay
produces a 2.6145 MeV photon that travels and converts (`Gamma`), the positron
annihilates a short distance away (`B`), and the resulting 511 keV photons
define a line of response (`P`) that is attenuated (`A`) and detected with
finite efficiency (`N`).

Every factor except `Gamma` is already present in a conventional PET
reconstruction, in the same place and with the same transpose. In PyTomography
terms `Gamma` is an **object-space transform**, applied to the image before the
projector, so it belongs in the `obj2obj_transforms` list of the system matrix.

Setting `Gamma = I` recovers conventional listmode PET, which converges to the
vertex density. That is the correct comparison arm: build both reconstructions
from the same events, geometry, subsets, iterations and initial image, and
change nothing but the presence of this transform.

## Minimal usage

```python
import hu_to_mu
from pp_vertex_mc import MCPPVertexTransform

bins = hu_to_mu.build_bins()
mu_tot, mu_pp = hu_to_mu.mu_maps(hu, 2.6145, bins)      # 1/mm, from CT

op = MCPPVertexTransform(mu_tot, mu_pp, b=0.358, n_photons=int(1e11), ndir=8)
op.setup(shape, dr)          # or op.configure(object_meta) under PyTomography
vertices = op.forward(activity_tensor)
```

Under PyTomography, `configure(object_meta)` is called for you and takes
`shape` and `dr` from the metadata, so `setup()` is only needed standalone.

## Invariants that must not be broken

These are the reasons the operator can be used inside an iterative algorithm at
all. An agent modifying this code should treat them as load-bearing.

1. **It is a fixed matrix, not a fresh estimate per application.** Three things
   make that true: emission is stratified rather than sampled from the image
   (photon group `g` starts in voxel `g % nvox` and carries the image value
   there as a weight); the path weight starts at unity so Russian roulette
   triggers at an image-independent depth; and `frozen=True` holds the seed
   fixed across subiterations. Do not set `frozen=False` in a reconstruction.

   This matters because the MLEM update is multiplicative. A matrix resampled
   at each application makes the iterate perform a random walk in log space,
   whose spread grows as `n^0.49` and dominates the image well before
   convergence.

2. **`backward()` is the exact transpose of what `forward()` realised.** It
   reuses the forward paths and gathers along them rather than scattering into
   them, so the two agree at any photon count, not merely in expectation.
   Re-emitting from the weighted source would be the transpose only in
   expectation and is wrong here.

3. **The operator is not self-adjoint.** `Gamma = b D K` with `D` diagonal in
   `mu_pp` and `K` a symmetric kernel, so `Gamma^T = b K D`, which differs from
   `Gamma` unless `mu_pp` is constant. `mu_pp` is evaluated at the collision
   site in both directions, which is what produces `KD` rather than `DK` in the
   transpose. Verify with an inner-product test on a heterogeneous map, never a
   uniform one, which cannot distinguish them.

4. **`n_photons` sets variance only.** The estimator is unbiased at any count.
   The quantity that matters is `K = n_photons / (ndir * nvox)`, the number of
   groups per voxel, exposed as `op.K`. Patient reconstructions in the paper
   used `n_photons = 1e11` with `ndir = 8` on a 128x128x131 grid, giving
   `K ~ 5.8e3`.

5. **A single energy is correct here, not an approximation bolted on.** The
   equation follows the primary photon to its first interaction and no further,
   `mu_tot` acting as a removal coefficient, so a photon that Compton scatters
   leaves the model at that point. No cross section away from the emission line
   is required. Conversions by already-scattered photons are outside the model;
   in the paper's patient studies those were 7.7% of GATE's vertices.

## Checks worth running after any change

```python
# adjoint, on a heterogeneous mu_pp map
y = torch.rand_like(f)
assert abs(float((op.forward(f)*y).sum()) - float((f*op.backward(y)).sum())) \
       / abs(float((op.forward(f)*y).sum())) < 1e-4

# linearity
assert torch.allclose(op.forward(2*f), 2*op.forward(f), rtol=1e-4)

# fixed matrix: same input, same output, bit for bit
assert torch.equal(op.forward(f), op.forward(f))
```

The adjoint residual is a single-precision accumulation floor and grows with
photon count: bit-exact at 5e7, around 1e-5 relative at 3e9. A residual that
grows faster than that, or a uniform-map test that passes while a
heterogeneous one fails, means invariant 2 or 3 has been broken.

## Files

- `pp_vertex_mc.py` — the operator; the only class you need is
  `MCPPVertexTransform`
- `pp_vertex_mc_cuda.py` — the Woodcock kernel, compiled at runtime by cupy
- `hu_to_mu.py` — CT to `mu_tot` / `mu_pp`; also writes a GATE material
  database if you need to cross-check against Monte Carlo
- `example.ipynb` — end to end on a synthetic CT

## Cross sections

`hu_to_mu.py` maps HU to density by the bilinear relation of Schneider et al.,
assigns one of five tissue classes by threshold, and subdivides each class into
density bins no wider than 0.05 g/cm3, giving 55 materials. Water values are
XCOM at the true line energy, scaled between classes by `<Z/A>` for the total
channel and `<Z^2/A>` for the pair channel, both from ICRP compositions.

The two scale differently because the processes do: between 1.5 and 3 MeV in
low-Z tissue the total coefficient is dominated by Compton scattering and
tracks electrons per gram, while the pair cross section goes as `Z^2` per atom.
Between adipose tissue and cortical bone `<Z/A>` differs by 7% and `<Z^2/A>` by
73%. That difference is why the operator depends on anatomy rather than acting
as a blur, and why a shift-invariant kernel cannot reproduce its behaviour
across a tissue boundary.
