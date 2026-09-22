#!/usr/bin/env python3
"""
Fit a plane to a wall captured by capture_wall.py and report how good it is.

    python3 fit_wall.py captures/wall_20260922_134150
    python3 fit_wall.py --latest
    python3 fit_wall.py --latest --tape-mm 1500

Reads points.npy, valid_mask.npy and meta.json from a capture directory,
fits the wall plane, and reports its distance, orientation and flatness.
With --tape-mm it checks the measured distance against a tape measure.


WHY RANSAC AND NOT LEAST SQUARES
--------------------------------
A least-squares fit has no idea which points are the wall. A skirting
board, a light switch, a stray false match on blank paint -- each drags
the plane toward itself. RANSAC finds the plane that the MOST points
agree with and ignores the rest, which is the definition of "the wall".
Least squares is then run on the RANSAC inliers only, because RANSAC
picks the right points and least squares gets the best plane through
them.


WHY THE CONSISTENCY MASK MATTERS HERE
-------------------------------------
capture_wall.py keeps a pixel only if it was valid in most of the burst
and its disparity barely moved. That throws away most of a blank wall's
interior -- SGBM cannot match featureless paint -- and keeps the edges,
the corners where it meets adjacent walls, and anything with texture.

That is the right trade for a plane fit. A plane needs a handful of
CORRECT points, spread out; it does not need dense coverage. What ruins
a plane fit is WRONG points, and interpolated or noisy disparities are
exactly that. So low coverage with high confidence beats full coverage
with guesses.

The one thing a plane fit does need is SPREAD. Points clustered along a
single edge constrain the plane poorly -- it can pivot about that edge.
The report checks the inlier footprint for that.


WHAT THE NUMBERS MEAN
---------------------
rms          how far inliers sit from the plane. The flatness of the
             reconstruction, not of the wall: a real wall is flat to a
             millimetre or two, so rms is almost entirely sensor noise.
             Expect a few mm at 1 m, growing with the square of range.

distance     perpendicular distance from the left lens to the wall.

tilt         angle between the wall's normal and the camera's viewing
             axis. 0 means the camera faces the wall square on.

expected     the depth noise predicted from the geometry, so rms can be
             judged against what the rig is capable of rather than
             against zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def ransac_plane(P, thresh, iters=1500, seed=0):
    """Dominant plane. Returns (normal, d, inlier_mask) or None."""
    rng = np.random.default_rng(seed)
    n = len(P)
    if n < 50:
        return None
    sub = P if n <= 8000 else P[rng.choice(n, 8000, replace=False)]

    best, best_count = None, 0
    for _ in range(iters):
        p0, p1, p2 = sub[rng.choice(len(sub), 3, replace=False)]
        nr = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nr)
        if ln < 1e-9:
            continue
        nr /= ln
        d = float(-nr @ p0)
        c = int((np.abs(sub @ nr + d) < thresh).sum())
        if c > best_count:
            best, best_count = (nr, d), c
    if best is None:
        return None

    nr, d = best
    inl = np.abs(P @ nr + d) < thresh
    if inl.sum() < 20:
        return None
    # Refit on inliers: RANSAC picked the points, least squares finds the
    # best plane through them.
    c = P[inl].mean(0)
    _, _, Vt = np.linalg.svd(P[inl] - c, full_matrices=False)
    nr = Vt[2]
    d = float(-nr @ c)
    # Orient toward the camera so distance comes out positive.
    if d < 0:
        nr, d = -nr, -d
    inl = np.abs(P @ nr + d) < thresh
    return nr, d, inl


def centre_point(pts, mask, half=20):
    """3D point under the image centre -- where the user is aiming."""
    H, W = mask.shape
    sl = (slice(H // 2 - half, H // 2 + half), slice(W // 2 - half, W // 2 + half))
    P = pts[sl][mask[sl]]
    P = P[np.isfinite(P).all(1) & (P[:, 2] > 0)]
    return np.median(P, axis=0) if len(P) >= 10 else None


def planes_by_size(P, thresh, max_planes=5, min_pts=200):
    """Sequential RANSAC: peel off planes largest first."""
    rest = P.copy()
    idx = np.arange(len(P))
    out = []
    for k in range(max_planes):
        if len(rest) < min_pts:
            break
        r = ransac_plane(rest, thresh, seed=k)
        if r is None:
            break
        nr, d, inl = r
        if inl.sum() < min_pts:
            break
        full = np.zeros(len(P), bool)
        full[idx[inl]] = True
        out.append((nr, d, full))
        rest, idx = rest[~inl], idx[~inl]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?", help="captures/wall_<timestamp>")
    ap.add_argument("--latest", action="store_true",
                    help="use the most recent capture in ./captures")
    ap.add_argument("--thresh-mm", type=float, default=None,
                    help="RANSAC inlier band. Defaults to 3x the expected "
                         "depth noise at the wall's distance")
    ap.add_argument("--biggest", action="store_true",
                    help="take the plane with the most points instead of "
                         "the one under the image centre")
    ap.add_argument("--tape-mm", type=float, default=None,
                    help="tape-measured distance to the wall, to check "
                         "the absolute scale")
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

    pts = np.load(cap / "points.npy")
    mask = np.load(cap / "valid_mask.npy")
    meta = json.loads((cap / "meta.json").read_text()) \
        if (cap / "meta.json").exists() else {}
    unit = meta.get("units", "mm")

    P = pts[mask].astype(np.float64)
    P = P[np.isfinite(P).all(1) & (P[:, 2] > 0)]
    print(f"capture   {cap.name}")
    print(f"points    {len(P)} valid "
          f"({100 * mask.mean():.1f}% of the image)")
    if len(P) < 100:
        sys.exit("too few points to fit a plane -- the wall needs texture, "
                 "or its edges need to be in frame")

    # Depth used to size the RANSAC band. The median of EVERY point is the
    # wrong choice: background seen through a doorway can sit 5-10 m away
    # and drag it far past the wall, inflating the band. The centre pixel
    # is the wall you are aiming at, so use that when it is available.
    cp0 = centre_point(pts, mask)
    zmed = float(cp0[2]) if cp0 is not None else float(np.median(P[:, 2]))

    # Expected depth noise at this range, from the rig's geometry. Quarter
    # pixel of disparity is a fair figure for SGBM after burst averaging.
    fB = None
    if "baseline" in meta and "Q" in meta:
        Q = np.array(meta["Q"])
        fx = float(Q[2, 3])
        fB = fx * float(meta["baseline"])
    expected = 0.25 * zmed * zmed / fB if fB else None

    thresh = a.thresh_mm or (max(3.0 * expected, 8.0) if expected else 20.0)
    # Which plane? NOT simply the biggest. RANSAC's biggest plane is
    # whichever surface has the most matched pixels, and a blank wall has
    # few while a textured desk in the foreground has many -- so the desk
    # wins and the "wall" comes out at desk distance. Measured on this rig:
    # a wall at 1700 mm reported as 749 mm for exactly this reason.
    #
    # You aim the centre box at the wall, so the wall is the plane the
    # centre pixel lies on. Find several planes, then pick that one.
    planes = planes_by_size(P, thresh)
    if not planes:
        sys.exit("no plane found")
    cp = centre_point(pts, mask)
    if cp is not None and not a.biggest:
        dists = [abs(float(cp @ nr_ + d_)) for nr_, d_, _ in planes]
        pick = int(np.argmin(dists))
        print(f"centre     {cp[2]:.0f} {unit} deep -- choosing the plane "
              f"it lies on")
        if len(planes) > 1:
            for i, (nr_, d_, inl_) in enumerate(planes):
                tag = "  <-- chosen" if i == pick else ""
                print(f"  plane {i+1}: {d_:7.0f} {unit}  {inl_.sum():6d} pts  "
                      f"centre off by {dists[i]:6.0f}{tag}")
        if dists[pick] > 3 * thresh:
            print(f"WARNING   no plane passes near the centre -- is the "
                  f"centre box on the wall?")
    else:
        pick = 0
        if cp is None:
            print("centre    no valid depth at the centre -- using the "
                  "largest plane instead. Aim the box at something textured.")
    nr, d, inl = planes[pick]
    Q_ = P[inl]
    dist = Q_ @ nr + d

    # Re-evaluate expected noise at the wall's actual depth, now that we
    # know which points are the wall. This is what rms is judged against.
    zwall = float(np.median(Q_[:, 2]))
    if fB:
        expected = 0.25 * zwall * zwall / fB
    rms = float(np.sqrt((dist ** 2).mean()))
    p95 = float(np.percentile(np.abs(dist), 95))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(nr[2])))))

    # Footprint: how spread out the inliers are across the wall. Points
    # along one edge only let the plane pivot about that edge.
    u = np.cross(nr, [0.0, 1.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(nr, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(nr, u)
    span_u = float(np.ptp(Q_ @ u))
    span_v = float(np.ptp(Q_ @ v))

    print(f"\n--- wall plane ---")
    print(f"distance   {d:8.1f} {unit}  (perpendicular, from the left lens)")
    print(f"tilt       {tilt:8.1f} deg (0 = facing the wall square on)")
    print(f"normal     [{nr[0]:+.3f} {nr[1]:+.3f} {nr[2]:+.3f}]")
    print(f"inliers    {inl.sum():8d} of {len(P)} "
          f"({100 * inl.mean():.0f}%) within {thresh:.1f} {unit}")
    print(f"\n--- flatness ---")
    print(f"rms        {rms:8.2f} {unit}")
    print(f"p95        {p95:8.2f} {unit}")
    if expected:
        print(f"expected   {expected:8.2f} {unit}  "
              f"(sensor noise predicted at the wall, {zwall:.0f} {unit})")
        ratio = rms / expected
        verdict = ("excellent -- at the sensor's limit" if ratio < 1.5 else
                   "good" if ratio < 3 else
                   "noisy -- worse than the geometry predicts")
        print(f"verdict    {verdict}  ({ratio:.1f}x expected)")

    print(f"\n--- coverage ---")
    print(f"footprint  {span_u:.0f} x {span_v:.0f} {unit}")
    if min(span_u, span_v) < 0.15 * max(span_u, span_v):
        print("WARNING   inliers lie almost along a line -- the plane can "
              "pivot about it. Get a second edge, or texture, into frame.")

    if a.tape_mm:
        err = d - a.tape_mm
        print(f"\n--- against the tape ---")
        print(f"tape       {a.tape_mm:8.1f} {unit}")
        print(f"measured   {d:8.1f} {unit}")
        print(f"error      {err:+8.1f} {unit}  "
              f"({100 * err / a.tape_mm:+.2f}%)")
        print("           tape distance must be PERPENDICULAR to the wall, "
              "from the left lens")

    out = cap / "plane.json"
    out.write_text(json.dumps(dict(
        normal=nr.tolist(), d=d, units=unit, rms=rms, p95=p95,
        tilt_deg=tilt, inliers=int(inl.sum()), points=int(len(P)),
        thresh=thresh, span=[span_u, span_v],
        expected_noise=expected, tape=a.tape_mm), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()