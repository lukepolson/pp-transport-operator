#!/usr/bin/env python3
"""CT Hounsfield units to attenuation coefficients and GATE materials.

Produces, from one CT volume, the two things the pipeline needs to stay
self-consistent:

  1. voxel maps of mu_tot and mu_pp at a given photon energy, in 1/mm, for
     the analytical L_mu operator;
  2. a GATE material database plus a `voxel_materials` table, for the Monte
     Carlo.

Both are generated from the *same* HU binning, which matters more than the
absolute accuracy of either.  GATE's ImageVolume assigns one material -- and
therefore one fixed density -- per HU interval, whereas a continuous
bilinear rho(HU) does not.  If the analytical maps used the continuous curve
while GATE used binned materials, the kernel-validation experiment would
measure that discrepancy rather than the physics it is meant to test.  So we
bin once, at `density_tolerance`, and both consumers read the same bins.

Model (deliberately simple -- this is a proof-of-principle paper):

  * rho(HU): the standard two-segment bilinear, Schneider et al 1996.
  * material class: HU thresholds into air / lung / adipose / soft tissue /
    spongiosa / cortical bone.
  * mass attenuation: scaled from NIST XCOM water values.  At 1.5-3 MeV in
    low-Z tissue, Compton dominates, so (mu/rho)_tot tracks <Z/A> (electrons
    per gram) and (mu/rho)_pp tracks <Z^2/A> (the pair cross section goes as
    Z^2 per atom).  Both ratios are computed from elemental composition.

The Z^2/A scaling for the pair channel is the physically interesting part:
cortical bone has ~1.6x the pair yield per gram of soft tissue *and* ~1.9x
the density, so PP vertices concentrate strongly in bone and are suppressed
in lung.  That spatial variation is precisely what a shift-invariant kernel
cannot represent.
"""

import argparse
import json
from pathlib import Path

import numpy as np

# --- Element data: Z, standard atomic mass ---------------------------------
ELEMENTS = {
    "H": (1, 1.008), "C": (6, 12.011), "N": (7, 14.007), "O": (8, 15.999),
    "Na": (11, 22.990), "Mg": (12, 24.305), "P": (15, 30.974), "S": (16, 32.06),
    "Cl": (17, 35.45), "Ar": (18, 39.948), "K": (19, 39.098), "Ca": (20, 40.078),
    "Fe": (26, 55.845), "Zn": (30, 65.38),
}

# --- Tissue classes: ICRP/ICRU elemental mass fractions ---------------------
# Compositions match the G4_*_ICRP NIST definitions so that the GATE database
# we emit is consistent with Geant4's own materials.
TISSUES = {
    "AIR": dict(
        g4="G4_AIR", rho_nominal=1.205e-3,
        comp={"C": 0.000124, "N": 0.755268, "O": 0.231781, "Ar": 0.012827}),
    "LUNG": dict(
        g4="G4_LUNG_ICRP", rho_nominal=1.05,
        comp={"H": 0.101278, "C": 0.10231, "N": 0.02865, "O": 0.757072,
              "Na": 0.00184, "Mg": 0.00073, "P": 0.0008, "S": 0.00225,
              "Cl": 0.00266, "K": 0.00194, "Ca": 0.00009, "Fe": 0.00037,
              "Zn": 0.00001}),
    # MISLABELLED: these mass fractions are ICRU Report 44 adipose tissue, not
    # ICRP.  NIST/ICRP (and therefore Geant4's G4_ADIPOSE_TISSUE_ICRP, and
    # SIMIND's adipose_icrp.cr4) is H 11.948 C 63.724 N 0.797 O 23.233.  The
    # two differ by -0.35% in mu_tot and +1.95% in mu_pp, which is the entire
    # adipose discrepancy reported in context/CROSS_SECTIONS.md.  Left as-is so
    # that existing runs stay reproducible; switch it deliberately, together
    # with the GATE database it feeds, or relabel it honestly as ICRU-44.
    "ADIPOSE": dict(
        g4="G4_ADIPOSE_TISSUE_ICRP", rho_nominal=0.95,
        comp={"H": 0.114, "C": 0.598, "N": 0.007, "O": 0.278,
              "Na": 0.001, "S": 0.001, "Cl": 0.001}),
    "SOFT": dict(
        g4="G4_TISSUE_SOFT_ICRP", rho_nominal=1.03,
        comp={"H": 0.104472, "C": 0.23219, "N": 0.02488, "O": 0.630238,
              "Na": 0.00113, "Mg": 0.00013, "P": 0.00133, "S": 0.00199,
              "Cl": 0.00134, "K": 0.00199, "Ca": 0.00023, "Fe": 0.00005,
              "Zn": 0.00003}),
    "SPONGIOSA": dict(
        g4="G4_B-100_BONE", rho_nominal=1.45,
        comp={"H": 0.065471, "C": 0.536945, "N": 0.0215, "O": 0.032085,
              "F": 0.167411, "Ca": 0.176589}),
    "BONE": dict(
        g4="G4_BONE_CORTICAL_ICRP", rho_nominal=1.92,
        comp={"H": 0.047234, "C": 0.14433, "N": 0.04199, "O": 0.446096,
              "Mg": 0.0022, "P": 0.10497, "S": 0.00315, "Ca": 0.20993,
              "Zn": 0.0001}),
}
# G4_B-100_BONE contains fluorine, which is not in ELEMENTS; drop that class
# rather than carry a partial composition.  Spongiosa is covered adequately by
# interpolating between SOFT and BONE via the density bins.
del TISSUES["SPONGIOSA"]

# HU intervals -> tissue class.  Upper bound exclusive; last is open-ended.
HU_CLASSES = [
    (-10000, -950, "AIR"),
    (-950,   -120, "LUNG"),
    (-120,    -20, "ADIPOSE"),
    (-20,     120, "SOFT"),
    (120,   10000, "BONE"),
]

# --- Reference water values, XCOM (cm^2/g) ----------------------------------
# Keyed by photon energy in MeV.  Read off the 0.5 keV XCOM grid shipped with
# SIMIND (smc_dir/h2o.cr4) via scripts/simind_xs.py -- NOT interpolated by hand
# from NIST's coarse output grid, which is how the previous values went wrong.
# Full provenance and validation against the published NIST tables:
# context/CROSS_SECTIONS.md.
#
# Use the true line energies.  2.6145 MeV is Tl-208 (Pb-212 chain) and 1.567
# MeV is Tl-209 (Ac-225 chain); the round 2.6 and 1.6 rows are kept only so
# that older runs remain reproducible.
#
# Two corrections landed here, both material:
#   * 1.6 MeV pair was 5.750e-5, which is water's value at 1396.5 keV -- low by
#     a factor 2.5.  Combined with the 33 keV energy error it understated the
#     Ac-225 PP branching ratio by 2.28x.
#   * 2.6 MeV tot was 4.29e-2, a linear interpolation of NIST's 0.5 MeV grid
#     across a convex curve, high by 0.13%.  The 0.0413 cm^-1 that circulates
#     in pp_vertex_transform.py is a different hand-interpolation error, low by
#     3.6%, and is not an attenuation coefficient for water at 2.6 MeV under
#     any channel combination.  Its agreement with the body-averaged mu_tot
#     over a patient CT (0.00413 /mm) is a coincidence.
WATER_MASS_ATTEN = {
    2.6145: dict(tot=4.271508e-2, pair=8.346591e-4),   # Tl-208, use this
    2.6:    dict(tot=4.284270e-2, pair=8.239717e-4),
    1.567:  dict(tot=5.625144e-2, pair=1.292488e-4),   # Tl-209, use this
    1.6:    dict(tot=5.564527e-2, pair=1.457408e-4),
    0.511:  dict(tot=9.598536e-2, pair=0.0),           # below the 1.022 threshold
}
WATER_RHO = 1.0


def _mean_z_over_a(comp):
    return sum(f * ELEMENTS[e][0] / ELEMENTS[e][1] for e, f in comp.items())


def _mean_z2_over_a(comp):
    return sum(f * ELEMENTS[e][0] ** 2 / ELEMENTS[e][1] for e, f in comp.items())


_WATER_COMP = {"H": 0.111894, "O": 0.888106}
_W_ZA = _mean_z_over_a(_WATER_COMP)
_W_Z2A = _mean_z2_over_a(_WATER_COMP)


def mass_atten(tissue, energy_mev):
    """(mu/rho)_tot and (mu/rho)_pair for a tissue class, in cm^2/g."""
    if energy_mev not in WATER_MASS_ATTEN:
        raise KeyError(
            f"No tabulated water mass-attenuation for {energy_mev} MeV; "
            f"have {sorted(WATER_MASS_ATTEN)}. Add a row from NIST XCOM.")
    w = WATER_MASS_ATTEN[energy_mev]
    comp = TISSUES[tissue]["comp"]
    return (w["tot"] * _mean_z_over_a(comp) / _W_ZA,
            w["pair"] * _mean_z2_over_a(comp) / _W_Z2A)


def hu_to_density(hu):
    """Two-segment bilinear rho(HU) in g/cm^3 (Schneider et al 1996).

    Below 0 HU the CT number is linear in density between air and water;
    above it the slope flattens as mineral content rises.  Clamped at zero so
    that CT padding values (GE writes -3024 HU outside the field of view,
    below physical air) cannot produce negative density.
    """
    hu = np.asarray(hu, dtype=np.float64)
    rho = np.where(hu <= 0.0, 1.0 + hu / 1000.0, 1.0 + hu * 5.67e-4)
    return np.clip(rho, 1.205e-3, None)


def classify(hu):
    """HU array -> index into HU_CLASSES."""
    out = np.full(np.shape(hu), len(HU_CLASSES) - 1, dtype=np.int8)
    for i, (lo, hi, _) in enumerate(HU_CLASSES):
        out[(hu >= lo) & (hu < hi)] = i
    return out


def build_bins(density_tolerance=0.05):
    """Subdivide each HU class into density bins no wider than the tolerance.

    Returns a list of dicts, each one HU interval with a single material name,
    a representative density, and its parent tissue class.  This is the single
    source of truth shared by the analytical maps and the GATE database.
    """
    bins = []
    for lo, hi, tissue in HU_CLASSES:
        # Clamp the open-ended outer classes to a physical CT range.
        lo_c, hi_c = max(lo, -1000), min(hi, 3000)
        if tissue == "AIR":
            bins.append(dict(hu_lo=lo, hu_hi=hi, tissue=tissue,
                             rho=TISSUES["AIR"]["rho_nominal"],
                             name="ppet_AIR"))
            continue
        r_lo = float(hu_to_density(lo_c))
        r_hi = float(hu_to_density(hi_c))
        n = max(1, int(np.ceil((r_hi - r_lo) / density_tolerance)))
        edges = np.linspace(lo_c, hi_c, n + 1)
        for k in range(n):
            h0, h1 = edges[k], edges[k + 1]
            rho = float(hu_to_density(0.5 * (h0 + h1)))
            bins.append(dict(
                hu_lo=float(h0) if k else float(lo),
                hu_hi=float(h1) if k < n - 1 else float(hi),
                tissue=tissue, rho=rho,
                name=f"ppet_{tissue}_{k:02d}"))
    return bins


def mu_maps(hu, energy_mev, bins):
    """mu_tot and mu_pp maps in 1/mm from an HU array, using the shared bins."""
    hu = np.asarray(hu, dtype=np.float64)
    mu_tot = np.zeros(hu.shape, dtype=np.float32)
    mu_pp = np.zeros(hu.shape, dtype=np.float32)
    for b in bins:
        sel = (hu >= b["hu_lo"]) & (hu < b["hu_hi"])
        if not sel.any():
            continue
        mt, mp = mass_atten(b["tissue"], energy_mev)
        # cm^2/g * g/cm^3 = 1/cm, then /10 for 1/mm.
        mu_tot[sel] = mt * b["rho"] / 10.0
        mu_pp[sel] = mp * b["rho"] / 10.0
    return mu_tot, mu_pp


def write_gate_material_db(bins, path):
    """Emit a GATE material database defining one material per density bin."""
    lines = ["# Auto-generated by scripts/hu_to_mu.py -- do not edit by hand.",
             "[Elements]"]
    used = sorted({e for b in bins for e in TISSUES[b["tissue"]]["comp"]})
    for e in used:
        z, a = ELEMENTS[e]
        lines.append(f"{e}: S= {e} ; Z= {z} ; A= {a} g/mole")
    lines.append("")
    lines.append("[Materials]")
    for b in bins:
        comp = TISSUES[b["tissue"]]["comp"]
        lines.append(f"{b['name']}: d={b['rho']:.6g} g/cm3 ; n={len(comp)}")
        for e, f in sorted(comp.items()):
            lines.append(f"        +el: name={e} ; f={f:.6g}")
    Path(path).write_text("\n".join(lines) + "\n")


def voxel_materials(bins):
    """GATE ImageVolume `voxel_materials` table: [[hu_lo, hu_hi, name], ...]."""
    return [[b["hu_lo"], b["hu_hi"], b["name"]] for b in bins]


def process(patient_dir, energy_mev, density_tolerance, out_name=None):
    import SimpleITK as sitk        # only the CLI path needs it
    patient_dir = Path(patient_dir)
    nifti = patient_dir / "nifti"
    ct = sitk.ReadImage(str(nifti / "ct.nii.gz"))
    hu = sitk.GetArrayFromImage(ct)

    bins = build_bins(density_tolerance)
    mu_tot, mu_pp = mu_maps(hu, energy_mev, bins)

    tag = out_name or f"{energy_mev:g}MeV"
    gate_dir = patient_dir / "gate"
    gate_dir.mkdir(exist_ok=True)

    for name, arr in (("mu_tot", mu_tot), ("mu_pp", mu_pp)):
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ct)
        sitk.WriteImage(img, str(nifti / f"{name}_{tag}.nii.gz"), True)

    # GATE reads the HU volume itself and maps it through voxel_materials.
    sitk.WriteImage(sitk.Cast(ct, sitk.sitkInt16), str(gate_dir / "ct_hu.mhd"))
    write_gate_material_db(bins, gate_dir / "ppet_materials.db")
    (gate_dir / "voxel_materials.json").write_text(
        json.dumps(voxel_materials(bins), indent=1))

    frac = {t: float((classify(hu) == i).mean())
            for i, (_, _, t) in enumerate(HU_CLASSES)}
    summary = dict(
        patient=patient_dir.name, energy_mev=energy_mev,
        density_tolerance=density_tolerance, n_material_bins=len(bins),
        volume_fraction_by_class={k: round(v, 4) for k, v in frac.items()},
        mu_tot_1_per_mm=dict(min=float(mu_tot.min()), max=float(mu_tot.max()),
                             mean_in_body=float(mu_tot[hu > -500].mean())),
        mu_pp_1_per_mm=dict(min=float(mu_pp.min()), max=float(mu_pp.max()),
                            mean_in_body=float(mu_pp[hu > -500].mean())),
    )
    (patient_dir / f"mu_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=r"C:\Users\lukep\Documents\Ga68-TCIA")
    p.add_argument("--patient", default=None, help="Default: all in --root")
    p.add_argument("--energy", type=float, default=2.6, help="Photon energy, MeV")
    p.add_argument("--density-tolerance", type=float, default=0.05,
                   help="Max density width per generated material, g/cm3")
    args = p.parse_args()

    root = Path(args.root)
    pts = ([root / args.patient] if args.patient
           else sorted(d for d in root.iterdir() if (d / "nifti").is_dir()))

    bins = build_bins(args.density_tolerance)
    print(f"Energy {args.energy} MeV, {len(bins)} material bins "
          f"(tolerance {args.density_tolerance} g/cm3)\n")
    for d in pts:
        s = process(d, args.energy, args.density_tolerance)
        mt, mp = s["mu_tot_1_per_mm"], s["mu_pp_1_per_mm"]
        print(f"  {d.name:24s} mu_tot body-mean {mt['mean_in_body']:.5f}/mm  "
              f"max {mt['max']:.5f}   mu_pp body-mean {mp['mean_in_body']:.3e}/mm  "
              f"max {mp['max']:.3e}")


if __name__ == "__main__":
    main()
