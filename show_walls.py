#!/usr/bin/env python3
"""
Draw the fitted walls and a floor grid over the live camera view.

    python3 show_walls.py --calib stereo_calib.npz --walls walls.npz

    G   grid on / off
    W   walls on / off
    D   live | disparity
    Q   quit

This is the check that no amount of top-down rendering can give you: if
the projected floor grid lies flat on the real floor and the wall lines
land on the real walls, the geometry is right. If they float, tilt or
slide, it is not -- and you can see which immediately.


WHY PROJECTION IS THE HONEST TEST
---------------------------------
A top-down plot shows you what the algorithm believes. Projecting that
belief back into the camera image tests it against what the camera
actually sees. Those are different claims, and only the second one can
fail visibly.


THE POSE PROBLEM, STATED PLAINLY
--------------------------------
The walls were fitted in the frame the head occupied when the sweep
started. To draw them now, this needs to know where the head is relative
to that frame -- and with the head moved, it does not.

So it assumes the head is back at the sweep's starting pose. Point the
rig roughly where it was when you pressed R and the overlay will line
up; turn away and it will not, and that is expected rather than a fault.

Live tracking would fix it, at the cost of re-running the odometry every
frame. With an encoder on the pan axis it becomes trivial: read the
angle, rotate the walls by it, draw.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import os as _os
_os.environ.setdefault("QT_LOGGING_RULES", "*=false")
_os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stereo_camera import CameraSettings, SyncedStereo      # noqa: E402


def build_sgbm(nd=128, bs=7):
    nd = int(np.ceil(nd / 16.0) * 16)
    left = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=bs,
        P1=8 * 3 * bs ** 2, P2=32 * 3 * bs ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=150, speckleRange=2, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    try:
        right = cv2.ximgproc.createRightMatcher(left)
        wls = cv2.ximgproc.createDisparityWLSFilter(left)
        wls.setLambda(8000.0); wls.setSigmaColor(1.5)
        return left, right, wls, nd
    except (AttributeError, cv2.error):
        return left, None, None, nd


def find_floor(xyz, valid, min_pts=2000):
    """
    Measure the floor from the stereo depth, rather than being told it.

    RANSAC over the lower half of the image, constrained to planes whose
    normal is within about 25 degrees of vertical. The lower-half
    restriction matters: the dominant plane in a whole frame is usually
    a WALL, and a wall satisfies "large planar surface" just as well as
    a floor does. Where it is in the image is the cheap discriminator.

    Returns (height_below_camera_mm, normal) or None when the floor is
    not in view -- in which case nothing is guessed, and the overlay
    simply says so.
    """
    h = valid.shape[0]
    lower = np.zeros_like(valid)
    lower[h // 2:, :] = True
    sel = valid & lower & np.isfinite(xyz[:, :, 2])
    P = xyz[sel]
    P = P[(P[:, 2] > 400) & (P[:, 2] < 4000)]
    if len(P) < min_pts:
        return None

    rng = np.random.default_rng(0)
    sub = P if len(P) <= 4000 else P[rng.choice(len(P), 4000, replace=False)]
    best, bc = None, 0
    for _ in range(300):
        p0, p1, p2 = sub[rng.choice(len(sub), 3, replace=False)]
        nr = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nr)
        if ln < 1e-6:
            continue
        nr /= ln
        if abs(nr[1]) < 0.9:                 # must be near-horizontal
            continue
        dd = float(-nr @ p0)
        c = int((np.abs(sub @ nr + dd) < 50.0).sum())
        if c > bc:
            best, bc = (nr, dd), c
    if best is None or bc < min_pts // 4:
        return None
    nr, dd = best
    inl = np.abs(P @ nr + dd) < 50.0
    c = P[inl].mean(0)
    _, _, Vt = np.linalg.svd(P[inl] - c, full_matrices=False)
    nr = Vt[2]
    if nr[1] < 0:
        nr = -nr                              # +Y is down in OpenCV
    return float(np.median(P[inl][:, 1])), nr


def project(pts3, P1):
    """mm in the camera frame -> pixels. Points behind the camera are
    dropped rather than wrapped around, which is what makes a line
    crossing the image edge look sane instead of exploding."""
    pts3 = np.asarray(pts3, np.float64).reshape(-1, 3)
    ok = pts3[:, 2] > 1.0
    uv = np.full((len(pts3), 2), np.nan)
    if ok.any():
        Q = pts3[ok]
        uv[ok, 0] = P1[0, 0] * Q[:, 0] / Q[:, 2] + P1[0, 2]
        uv[ok, 1] = P1[1, 1] * Q[:, 1] / Q[:, 2] + P1[1, 2]
    return uv, ok


def draw_polyline(img, pts3, P1, colour, thickness=1):
    uv, ok = project(pts3, P1)
    for i in range(len(uv) - 1):
        if not (ok[i] and ok[i + 1]):
            continue
        a = tuple(np.round(uv[i]).astype(int))
        b = tuple(np.round(uv[i + 1]).astype(int))
        cv2.line(img, a, b, colour, thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="stereo_calib.npz")
    ap.add_argument("--walls", default="walls.npz",
                    help="from build_grid's saved walls")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--floor-mm", type=float, default=None,
                    help="height of the floor below the cameras. Left "
                         "unset it is MEASURED from the stereo depth "
                         "every frame, which is the whole point of "
                         "having a depth sensor.")
    ap.add_argument("--grid-mm", type=float, default=500.0,
                    help="spacing of the floor grid")
    ap.add_argument("--grid-extent-mm", type=float, default=3000.0)
    ap.add_argument("--wall-height-mm", type=float, default=2400.0)
    a = ap.parse_args()

    with np.load(a.calib) as f:
        K1, D1, K2, D2 = f["K1"], f["D1"], f["K2"], f["D2"]
        R, T = f["R"], f["T"]
    size = (a.width, a.height)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
    m2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2)
    baseline = abs(float(np.asarray(T).ravel()[0]))
    print(f"[geom] fx {P1[0,0]:.1f}  baseline {baseline:.2f} mm")

    walls = None
    wp = Path(a.walls)
    if wp.exists():
        walls = np.load(wp)["walls"]      # (N,4): ax, az, bx, bz in mm
        print(f"[walls] {len(walls)} loaded")
    else:
        print(f"[walls] {wp} not found -- grid only")

    cam = SyncedStereo(settings=CameraSettings(width=a.width,
                                               height=a.height))
    cam.start()

    matcher, right_matcher, wls, num_disp = build_sgbm()
    min_disp = max(1.0, P1[0, 0] * baseline / 6000.0)

    print("\nG grid | W walls | D view | Q quit")
    print("Point the rig where it was when the sweep started.\n")
    cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
    show_grid, show_walls, view = True, True, 0
    yoff = a.floor_mm if a.floor_mm is not None else 1200.0
    floor_locked = a.floor_mm is not None
    floor_seen = False

    # Pre-build the floor grid once: it never changes.
    e, s = a.grid_extent_mm, a.grid_mm
    lines = []
    for x in np.arange(-e, e + 1, s):
        lines.append(np.array([[x, yoff, z] for z in np.arange(0, e + 1, 100)]))
    for z in np.arange(0, e + 1, s):
        lines.append(np.array([[x, yoff, z] for x in np.arange(-e, e + 1, 100)]))

    try:
        while True:
            left, right = cam.capture()
            rl = cv2.remap(left, m1[0], m1[1], cv2.INTER_LINEAR)
            rr = cv2.remap(right, m2[0], m2[1], cv2.INTER_LINEAR)
            gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
            raw = matcher.compute(gl, gr)
            if wls is not None:
                raw = wls.filter(raw, rl, None, right_matcher.compute(gr, gl))
            disp = raw.astype(np.float32) / 16.0
            valid = (disp >= min_disp) & (disp <= num_disp - 1.0)
            xyz = cv2.reprojectImageTo3D(
                np.where(valid, disp, 1.0).astype(np.float32), Q)

            if not floor_locked:
                f = find_floor(xyz, valid)
                if f is not None:
                    # Smoothed: a per-frame estimate jitters by tens of mm
                    # and a grid that twitches is harder to judge than one
                    # that is slightly wrong.
                    yoff = 0.85 * yoff + 0.15 * f[0] if floor_seen else f[0]
                    floor_seen = True

            shown = rl.copy() if view == 0 else cv2.applyColorMap(
                np.clip(disp / max(num_disp * 0.6, 1) * 255, 0, 255
                        ).astype(np.uint8), cv2.COLORMAP_TURBO)

            if show_grid:
                for L in lines:
                    draw_polyline(shown, L, P1, (0, 160, 0), 1)

            if show_walls and walls is not None:
                for i, w in enumerate(walls):
                    ax, az, bx, bz = w[:4]
                    # Each wall as a vertical quad from floor upward.
                    base = np.array([[ax, yoff, az], [bx, yoff, bz]])
                    top = base.copy()
                    top[:, 1] = yoff - a.wall_height_mm
                    col = (0, 220, 255)
                    draw_polyline(shown, base, P1, col, 2)
                    draw_polyline(shown, top, P1, col, 1)
                    for p, q in zip(base, top):
                        draw_polyline(shown, np.array([p, q]), P1, col, 1)

            txt = [f"grid {'on' if show_grid else 'off'} "
                   f"({a.grid_mm:.0f} mm)   "
                   f"walls {'on' if show_walls else 'off'}   "
                   f"skew {cam.skew_ms*1000:.0f} us",
                   (f"floor MEASURED at {yoff:.0f} mm below the cameras"
                    if floor_seen and not floor_locked else
                    (f"floor fixed at {yoff:.0f} mm" if floor_locked else
                     "floor NOT VISIBLE -- tilt down until it is"))]
            for i, t in enumerate(txt):
                for th, c in ((3, (0, 0, 0)), (1, (0, 255, 0))):
                    cv2.putText(shown, t, (8, 20 + i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, th,
                                cv2.LINE_AA)

            cv2.imshow("overlay", shown)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("g"):
                show_grid = not show_grid
            elif k == ord("w"):
                show_walls = not show_walls
            elif k == ord("d"):
                view = 1 - view
            elif k == ord("="):
                yoff += 50
            elif k == ord("-"):
                yoff -= 50
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()