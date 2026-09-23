#!/usr/bin/env python3
"""
Fit the wall in a capture_wall.py capture and measure how flat it is.

    python3 fit_wall.py --latest
    python3 fit_wall.py captures/wall_20260922_175530
    python3 fit_wall.py --latest --tape-mm 2780

Reads disparity_mean.npy, valid_mask.npy, meta.json and left_rect.png.
Writes plane.json (used by make_grid.py to level the grid) and
residual.png (distance of every point from the plane, over the image).
If no trustworthy wall is found, nothing is written and any old
plane.json is removed, so make_grid.py cannot level on a bad plane.


WHICH PLANE
-----------
Exactly the fitting live_grid.py uses -- imported from it, so the two
cannot drift apart. With steady depth under the image centre, the plane
through that point. With blank paint there, the plane whose points
surround the centre, built from the wall's edges. Candidates closer than
the rig can resolve, tilted beyond --max-tilt, lying along one edge only,
or joining two unrelated surfaces are rejected with the reason.


WHAT THE FLATNESS NUMBERS MEAN, AND WHEN THEY MEAN NOTHING
----------------------------------------------------------
band         tolerance for "on the plane": 3x the depth noise predicted at
             the wall's distance. Sized from the wall, never from a single
             pixel.
rms / p95    how far the on-plane points sit from it. On a real wall this
             is sensor noise: walls are flat to a few mm, the rig resolves
             about +-14 mm at 1.7 m. So "at the sensor's limit" means
             flatter than the rig can see, not a measured flatness.
kept         of the points near the plane (within 5 bands), the fraction
             inside the band. A flat wall keeps most; a bulge or step
             bigger than the band shows up as a drop here, NOT in rms --
             rms only ever describes the points that were kept.
band-filling if rms exceeds half the band, the points are spread evenly
             across the tolerance slab rather than clustered on a plane
             (uniform spread gives band/sqrt(3)). That is not a plane, and
             the verdict is withheld.
residual.png every steady point coloured by its distance from the plane:
             green on it, red nearer, blue farther. Evenly spaced stripes
             running along lines of equal distance are SGBM's sub-pixel
             quantization (disparities bunch up near whole pixels), a few
             mm, present on any surface -- not waviness in the wall.
bow          a smooth quadratic fitted to the residuals across the wall,
             judged only when the points cover at least 70% of it -- fitted
             to a ring of edge points it is unreliable.
             A local bump means the wall; a symmetric bowl or saddle
             centred on the image means the CALIBRATION -- a wrong
             distortion model or optical centre bends a flat surface that
             way. On a wide, textured, genuinely flat surface (the floor)
             a bow within the noise independently confirms the
             calibration, including the ~80 px cy offset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_grid import fit_frame, tilt_deg                       # noqa: E402


def in_plane_axes(nr):
    u = np.cross(nr, [0.0, 1.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(nr, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    return u, np.cross(nr, u)


def bow(uv, resid):
    """Quadratic part of a smooth surface fitted to the residuals: its
    largest deviation across the footprint, in mm. Plane terms are removed
    because the plane fit has already absorbed them."""
    u, v = uv[:, 0], uv[:, 1]
    su, sv = max(np.ptp(u), 1.0), max(np.ptp(v), 1.0)
    un, vn = (u - u.mean()) / su, (v - v.mean()) / sv
    A = np.stack([np.ones_like(un), un, vn, un * un, un * vn, vn * vn], 1)
    coef = np.linalg.lstsq(A, resid, rcond=None)[0]
    quad = A[:, 3:] @ coef[3:]
    return float(np.ptp(quad)), coef[3:]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?", help="captures/wall_<timestamp>")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--max-tilt", type=float, default=55.0)
    ap.add_argument("--min-inliers", type=int, default=300)
    ap.add_argument("--verbose", action="store_true",
                    help="list every candidate plane and why it was accepted "
                         "or rejected")
    ap.add_argument("--tape-mm", type=float, default=None,
                    help="tape-measured PERPENDICULAR distance from the left "
                         "lens to the wall")
    a = ap.parse_args()

    if a.latest:
        caps = sorted(Path("captures").glob("wall_*"))
        if not caps:
            sys.exit("no captures in ./captures")
        cap = caps[-1]
    elif a.capture:
        cap = Path(a.capture)
    else:
        sys.exit("give a capture directory, or --latest")

    meta = json.loads((cap / "meta.json").read_text())
    Q = np.array(meta["Q"], float)
    num_disp = int(meta.get("num_disp", 96))
    unit = meta.get("units", "mm")
    fB = float(Q[2, 3]) * float(meta["baseline"])
    z_min = fB / num_disp
    disp = np.load(cap / "disparity_mean.npy").astype(np.float32)   # offset applied
    mask = np.load(cap / "valid_mask.npy")
    plane_file = cap / "plane.json"

    print(f"capture   {cap.name}")
    print(f"steady    {mask.mean() * 100:.1f}% of the image   "
          f"nearest measurable {z_min:.0f} {unit}")

    log = []
    fit, note = fit_frame(disp, mask, Q, fB, num_disp, a.max_tilt,
                          a.min_inliers, np.random.default_rng(0), log=log)
    if a.verbose:
        print("\n--- candidates ---")
        for line in log:
            print("  " + line)
        print()
    if fit is None:
        if plane_file.exists():
            plane_file.unlink()
            print("removed   stale plane.json -- make_grid.py will not level on it")
        sys.exit(f"NO WALL   {note}")
    nr, d = np.asarray(fit[0], np.float64), float(fit[1])
    if note:
        print(f"method    {note}")

    # every steady point in the measurable range, and its distance from the plane
    xyz = cv2.reprojectImageTo3D(np.where(mask, disp, 1.0).astype(np.float32), Q)
    ys, xs = np.nonzero(mask)
    P = xyz[ys, xs].astype(np.float64)
    ok = np.isfinite(P).all(1) & (P[:, 2] > z_min) & (P[:, 2] < 6000)
    P, ys, xs = P[ok], ys[ok], xs[ok]
    resid = P @ nr + d

    axis = float(-d / nr[2])
    expected = float(0.25 * axis * axis / fB)   # quarter-pixel disparity error
    band = float(max(3.0 * expected, 8.0))
    inl = np.abs(resid) < band
    near = np.abs(resid) < 5 * band
    Qi, ri = P[inl], resid[inl]
    rms = float(np.sqrt((ri ** 2).mean()))
    p95 = float(np.percentile(np.abs(ri), 95))
    kept = float(inl.sum() / max(near.sum(), 1))
    band_filling = bool(rms > 0.5 * band)

    u, v = in_plane_axes(nr)
    uv = np.stack([Qi @ u, Qi @ v], 1)
    span_u, span_v = float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1]))
    bow_mm, _ = bow(uv, ri)
    # A curved surface fitted to points only around the edges is unreliable:
    # judge bow only when the points actually cover the footprint.
    g = 6
    cu = np.clip(((uv[:, 0] - uv[:, 0].min()) / max(span_u, 1.0) * g).astype(int), 0, g - 1)
    cv = np.clip(((uv[:, 1] - uv[:, 1].min()) / max(span_v, 1.0) * g).astype(int), 0, g - 1)
    fill = len(np.unique(cu * g + cv)) / (g * g)

    print("\n--- wall plane ---")
    print(f"axis       {axis:8.1f} {unit}  (along the viewing axis)")
    print(f"distance   {abs(d):8.1f} {unit}  (perpendicular, from the left lens)")
    print(f"tilt       {tilt_deg(nr):8.1f} deg (0 = square on)")
    print(f"normal     [{nr[0]:+.3f} {nr[1]:+.3f} {nr[2]:+.3f}]")

    print("\n--- flatness ---")
    print(f"band       {band:8.1f} {unit}  (3x the {expected:.1f} {unit} noise "
          f"predicted at {axis:.0f} {unit})")
    print(f"points     {int(inl.sum())} on the plane, {kept * 100:.0f}% of those near it")
    print(f"rms        {rms:8.2f} {unit}")
    print(f"p95        {p95:8.2f} {unit}")
    print(f"bow        {bow_mm:8.2f} {unit}  across {span_u:.0f} x {span_v:.0f} {unit}")
    if band_filling:
        verdict = ("WITHHELD -- points fill the tolerance band evenly "
                   f"(rms {rms / band:.2f} x band), so this is not a clean plane")
    else:
        ratio = rms / expected
        verdict = ("at the sensor's limit -- flatter than the rig can resolve"
                   if ratio < 1.5 else "good" if ratio < 3
                   else "noisy -- worse than the geometry predicts")
        if kept < 0.7:
            verdict += f"; only {kept * 100:.0f}% kept -- a step or bulge beyond the band"
    print(f"verdict    {verdict}")
    if band_filling:
        pass
    elif fill < 0.7:
        print(f"bow        not judged -- points cover {fill * 100:.0f}% of the footprint "
              f"(edges only); needs a textured surface, e.g. the floor")
    else:
        print("bow        " + ("within the noise -- no measurable curvature"
                               if bow_mm < expected else
                               "LARGER than the noise -- a smooth bowl or saddle "
                               "centred on the image points at the calibration; "
                               "a local bump points at the wall"))
    if min(span_u, span_v) < 0.15 * max(span_u, span_v):
        print("WARNING    points lie almost along a line -- the plane can pivot")

    if a.tape_mm:
        err = abs(d) - a.tape_mm
        print("\n--- against the tape ---")
        print(f"tape       {a.tape_mm:8.1f} {unit}   measured {abs(d):8.1f} {unit}   "
              f"error {err:+.1f} {unit} ({100 * err / a.tape_mm:+.2f}%)")

    # residual map. The normal faces the camera, so a POSITIVE residual is
    # nearer than the plane and a negative one is farther.
    base = cv2.imread(str(cap / "left_rect.png"))
    if base is None:
        base = np.zeros(mask.shape + (3,), np.uint8)
    img = (cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY),
                        cv2.COLOR_GRAY2BGR) * 0.5).astype(np.uint8)
    show = near
    t = np.clip(resid[show] / band, -1, 1)
    col = np.zeros((show.sum(), 3), np.uint8)
    col[:, 2] = np.clip(255 * t, 0, 255)                     # red: nearer
    col[:, 0] = np.clip(255 * -t, 0, 255)                    # blue: farther
    col[:, 1] = (255 * (1 - np.abs(t))).astype(np.uint8)     # green: on the plane
    img[ys[show], xs[show]] = col
    cv2.putText(img, f"green on plane | red nearer, blue farther (+-{band:.0f} {unit})",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(cap / "residual.png"), img)

    plane_file.write_text(json.dumps(dict(
        normal=nr.tolist(), d=float(d), axis_mm=float(axis), units=unit,
        tilt_deg=tilt_deg(nr), method=note or "centre",
        band=band, expected_noise=expected, rms=rms, p95=p95,
        inliers=int(inl.sum()), kept_fraction=float(kept),
        band_filling=bool(band_filling), bow_mm=bow_mm,
        bow_judged=bool(fill >= 0.7 and not band_filling), coverage=float(fill),
        span=[span_u, span_v], tape=a.tape_mm), indent=2))
    print(f"\nwrote {plane_file} and {cap / 'residual.png'}")


if __name__ == "__main__":
    main()