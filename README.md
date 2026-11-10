# Pair-production transport operator

A Monte Carlo photon-transport operator for PET imaging of alpha-emitter
daughters, and the CT conversion that feeds it.

Some alpha-therapy decay chains emit a gamma above the 1.022 MeV
pair-production threshold. A fraction of those photons convert to an
electron-positron pair inside the patient, the positron annihilates, and the
resulting 511 keV pair is recorded as a coincidence on a conventional PET
scanner. For 212Pb the photon is the 2.6145 MeV emission of 208Tl.

The catch is that conversion does not happen where the decay did. A 2.6145 MeV
photon travels a mean free path of about 24 cm in soft tissue, so a PET scanner
images the distribution of *conversion sites*, not the activity. This operator

```
  Gamma f(r) = b * mu_pp(r) * INT exp(-tau(r,r')) / (4 pi |r-r'|^2) f(r') d3r'
```

maps the activity distribution f to the distribution of photon-induced
electron-positron pairs (EPPs) through the patient's own CT. Placed in the system matrix as `H = N A P B Gamma`, it lets listmode
OSEM converge to the activity instead.

The excited 208Pb nucleus can also create the pair itself, by internal pair
formation, in `alpha_pi = 3.71e-4` of the transitions. Those pairs appear at the
decay site with no photon to transport, so the operator used in the paper is
`Gamma' = Gamma + b alpha_pi I`, set with `internal=alpha_pi`.

Release `pmb-r1` is the version that produced the results of the revised paper.

## Requirements

- An NVIDIA GPU. The tracker is a CUDA kernel and there is no CPU path.
- `numpy`, `torch` (CUDA build), `cupy`, `matplotlib`

`pytomography` is optional. When present, the operator subclasses
`pytomography.transforms.Transform`; when absent it falls back to a standalone
class with the same interface. `SimpleITK` is needed only for the CT-reading
command line path, not for the library.

```bash
pip install -r requirements.txt
```

## Quick start

```python
import hu_to_mu
from pp_vertex_mc import MCPPVertexTransform

bins = hu_to_mu.build_bins()
mu_tot, mu_pp = hu_to_mu.mu_maps(hu, 2.6145, bins)     # 1/mm, from your CT

op = MCPPVertexTransform(mu_tot, mu_pp, b=0.398, n_photons=int(1e9), ndir=8,
                         internal=3.71e-4)
op.setup(shape, dr)                                     # dr in mm
vertices = op.forward(activity)                         # torch tensor in, out
```

`example.ipynb` runs this end to end on a synthetic CT spanning all five tissue
classes, shows coronal slices of the CT, `mu_pp`, the activity and the vertex
density, and checks the adjoint. It takes a few seconds on a modern GPU.

## Using your own CT

Replace the synthetic HU array in the notebook with a real one and set `DR` to
its voxel size in mm:

```python
import SimpleITK as sitk
img = sitk.ReadImage('ct.nii.gz')
hu  = sitk.GetArrayFromImage(img)      # z, y, x
DR  = img.GetSpacing()[::-1]           # mm
```

Everything downstream is unchanged. `hu_to_mu.py` also has a command line entry
point that reads a study directory and writes the mu maps and a GATE material
database, if you want to cross-check against Monte Carlo.

## Choosing n_photons

`n_photons` sets the variance and nothing else; the estimator is unbiased at
any count. The quantity that matters is `K = n_photons / (ndir * nvox)`, the
number of photon groups launched per voxel, available as `op.K`. A few hundred
is enough to look at; the paper's patient reconstructions used `K` of about
5800.

## Notes

- The operator is a **fixed matrix** once `setup()` has run, not a fresh
  estimate at each application. Keep `frozen=True` inside a reconstruction:
  the MLEM update is multiplicative, and a resampled matrix makes the iterate
  random-walk in log space.
- `backward()` is the **exact transpose** of what `forward()` realised, at any
  photon count, because it gathers along the stored paths.
- The operator is **not self-adjoint** unless `mu_pp` is constant.
- `positron` (off by default, and off in the paper) scores the annihilation
  point instead of the conversion vertex, from a table of positron
  displacements.

`CLAUDE.md` has the integration details, the invariants, and the checks to run
after modifying anything.

## Citation

If you use this, please cite the paper describing the operator, L A Polson and
Y Seo, *Assessing the pair-production pathway for PET imaging of 212Pb with a
photon transport operator* (Physics in Medicine & Biology, under review), and
the archived release pmb-r1, [doi:10.5281/zenodo.24096097](https://doi.org/10.5281/zenodo.24096097). The analysis pipeline that produced the
paper's results is at
[lukepolson/pp-pet-212pb](https://github.com/lukepolson/pp-pet-212pb).

## License

MIT; see `LICENSE`.
