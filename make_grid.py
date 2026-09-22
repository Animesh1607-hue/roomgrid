#!/usr/bin/env python3
"""
Break a capture into a 3D occupancy grid.

    python3 make_grid.py --latest
    python3 make_grid.py captures/wall_20260922_164410 --voxel-mm 20
    python3 make_grid.py --latest --no-level

Every cell ends up in one of three states:

    OCCUPIED   a measured point landed in it -- something is there
    FREE       the camera saw THROUGH it to something behind -- empty
    UNKNOWN    never observed

Free space is the part that matters for manipulation. Knowing where the
objects are tells an arm what to reach for; knowing where the empty air
is tells it which path is safe. A grid of occupied cells alone cannot
distinguish "empty" from "never looked".


THE FRAME
---------
Origin at the midpoint of the baseline, between the two lenses. The
stereo points arrive relative to the LEFT lens, so they are shifted half
a baseline in X first.

Then, if a plane.json from fit_wall.py is present, LEVELLED: rotated so
the wall's normal is horizontal. The floor is not in view, but a wall is
vertical, so its normal lies in the horizontal plane -- and whatever
vertical component it shows in camera coordinates is the camera's pitch.
Rotating that away gives a gravity-aligned grid, which is what "up" has
to mean for a robot. Axes after levelling:

    +X  right      +Y  down      +Z  forward (horizontal)

Levelling assumes the wall really is vertical. It is a far safer
assumption than any other available one, but a leaning wall will lean
the grid by the same amount.


HOW FREE SPACE IS FOUND
-----------------------
For every measured point, the straight line from the camera to it passed
through empty air -- that is how the camera saw the point at all. So
every cell along that ray, short of the point itself, is free. Cells are
marked by sampling each ray at half-voxel steps.

A cell hit by a point AND crossed by other rays stays OCCUPIED: a point
is a positive observation, a ray passing nearby is not.


WHY THE CELL SIZE HAS A FLOOR
-----------------------------
Depth noise grows with the square of range. After burst averaging and
the disparity-offset correction this rig measures a wall to roughly:

     1.0 m   +/- 5 mm
     1.7 m   +/- 14 mm
     2.5 m   +/- 29 mm

A cell smaller than the noise at its range is not finer resolution, it
is a guess spread across several cells. 20 mm cells are honest to about
2 m; beyond that the grid is still correct but coarser than it looks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def level_rotation(normal):
    """
    Rotation that makes a wall normal horizontal, removing camera pitch.

    Rotates about the camera's X axis only. That removes pitch without
    introducing any yaw, so +Z stays pointing where the camera looked.
    """
    n = np.asarray(normal, float)
    # Pitch is the angle of the normal out of the horizontal (X-Z) plane.
    pitch = np.arctan2(n[1], np.hypot(n[0], n[2]))
    c, s = np.cos(-pitch), np.sin(-pitch)
    # Sign chosen so the rotated normal ends with zero Y component.
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if abs((R @ n)[1]) > abs((R.T @ n)[1]):
        R = R.T
    return R, float(np.degrees(pitch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--voxel-mm", type=float, default=20.0)
    ap.add_argument("--extent-mm", type=float, default=3000.0,
                    help="half-width of the grid in X and Y; Z runs 0 to this")
    ap.add_argument("--max-range-mm", type=float, default=4000.0,
                    help="ignore points further than this -- background "
                         "seen through doorways is unreliable and not useful")
    ap.add_argument("--no-level", action="store_true",
                    help="keep camera axes instead of levelling on the wall")
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
    baseline = float(meta.get("baseline", 64.64))

    P = pts[mask].astype(np.float64)
    ok = np.isfinite(P).all(1) & (P[:, 2] > 0) & \
        (np.linalg.norm(P, axis=1) < a.max_range_mm)
    P = P[ok]
    print(f"capture   {cap.name}")
    print(f"points    {len(P)} within {a.max_range_mm:.0f} mm")
    if len(P) < 100:
        sys.exit("too few points")

    # Left lens -> baseline midpoint. The camera origin moves with it.
    P[:, 0] -= baseline / 2.0
    cam = np.array([-baseline / 2.0, 0.0, 0.0])

    R = np.eye(3)
    pitch = 0.0
    plane_f = cap / "plane.json"
    if not a.no_level and plane_f.exists():
        pl = json.loads(plane_f.read_text())
        R, pitch = level_rotation(pl["normal"])
        P = P @ R.T
        cam = R @ cam
        print(f"levelled  removed {pitch:+.1f} deg of camera pitch "
              f"(from the wall normal)")
    elif not a.no_level:
        print("levelled  NO -- run fit_wall.py first to level on the wall")

    v = a.voxel_mm
    ex = a.extent_mm
    nx = ny = int(2 * ex / v)
    nz = int(ex / v)
    grid = np.zeros((nx, ny, nz), np.uint8)
    print(f"grid      {nx} x {ny} x {nz} cells of {v:.0f} mm "
          f"({grid.nbytes / 1e6:.1f} MB)")

    def to_idx(Q):
        ix = ((Q[:, 0] + ex) / v).astype(np.int64)
        iy = ((Q[:, 1] + ex) / v).astype(np.int64)
        iz = (Q[:, 2] / v).astype(np.int64)
        good = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                & (iz >= 0) & (iz < nz))
        return ix[good], iy[good], iz[good]

    # ---- free space: sample every ray from the camera to its point.
    # Stop half a voxel short so the endpoint cell is never marked free.
    step = v / 2.0
    rng = np.random.default_rng(0)
    rays = P if len(P) <= 60000 else P[rng.choice(len(P), 60000, replace=False)]
    dvec = rays - cam
    L = np.linalg.norm(dvec, axis=1)
    u = dvec / L[:, None]
    nmax = int(np.ceil(L.max() / step))
    for k in range(1, nmax):
        t = k * step
        live = t < (L - v * 0.5)
        if not live.any():
            break
        S = cam + u[live] * t
        ix, iy, iz = to_idx(S)
        grid[ix, iy, iz] = FREE

    # ---- occupied: positive observations override free.
    ix, iy, iz = to_idx(P)
    grid[ix, iy, iz] = OCCUPIED

    n_occ = int((grid == OCCUPIED).sum())
    n_free = int((grid == FREE).sum())
    n_unk = int((grid == UNKNOWN).sum())
    tot = grid.size
    print(f"\noccupied  {n_occ:9d}  ({100 * n_occ / tot:5.2f}%)")
    print(f"free      {n_free:9d}  ({100 * n_free / tot:5.2f}%)")
    print(f"unknown   {n_unk:9d}  ({100 * n_unk / tot:5.2f}%)")

    out = cap / "grid.npz"
    np.savez_compressed(
        out, grid=grid, voxel_mm=np.array(v), extent_mm=np.array(ex),
        origin=np.array([-ex, -ex, 0.0]),
        level_R=R, pitch_deg=np.array(pitch),
        states=np.array(["unknown", "free", "occupied"]))
    print(f"\nwrote {out}")

    # ---- pictures: top-down and side, so the grid can be checked by eye.
    try:
        import cv2
    except ImportError:
        return
    COL = {UNKNOWN: (30, 30, 30), FREE: (70, 110, 70), OCCUPIED: (0, 200, 255)}

    def render(state2d, name, xlabel, ylabel):
        h, w = state2d.shape
        img = np.zeros((h, w, 3), np.uint8)
        for st, c in COL.items():
            img[state2d == st] = c
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"{name}  {xlabel} across, {ylabel} up the page",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        cv2.putText(img, "yellow occupied  green free  grey unknown",
                    (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1)
        return img

    # Top-down: collapse Y (height). A column is occupied if any cell is,
    # else free if any cell is, else unknown.
    top = np.where((grid == OCCUPIED).any(1), OCCUPIED,
                   np.where((grid == FREE).any(1), FREE, UNKNOWN))
    cv2.imwrite(str(cap / "grid_top.png"),
                render(np.flipud(top.T), "TOP-DOWN", "X", "Z (forward)"))
    # Side: collapse X.
    side = np.where((grid == OCCUPIED).any(0), OCCUPIED,
                    np.where((grid == FREE).any(0), FREE, UNKNOWN))
    cv2.imwrite(str(cap / "grid_side.png"),
                render(side, "SIDE", "Z (forward)", "height (up is up)"))
    print(f"wrote {cap / 'grid_top.png'} and grid_side.png")


if __name__ == "__main__":
    main()