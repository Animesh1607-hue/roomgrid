#!/usr/bin/env python3
"""
Live, distance-accurate gridlines drawn onto the wall in the camera view.

    python3 live_grid.py
    python3 live_grid.py --spacing-mm 100 --extent-mm 1500

    +/-  change grid spacing
    L    lock / unlock the plane (freeze it once it looks right)
    R    reset the history -- use after moving the rig
    Q    quit

Point the centre box at the wall. Every few frames the wall plane is
refitted and a square grid is laid on it and drawn back into the image.
Because the grid lives on the measured plane in millimetres, each line
really is --spacing-mm from the next, at whatever distance and angle the
wall happens to be.


WHY THE SETTINGS MATCH capture_wall.py EXACTLY
----------------------------------------------
Same matcher parameters, NO WLS filter, and the same disparity offset.
The 1.15 px offset was measured and validated against a tape at two
distances -- +0.89% at 1.7 m, -0.14% at 2.5 m -- with those exact
settings. WLS changes disparities, so turning it on here would silently
invalidate the correction and put every gridline in the wrong place.


WHY IT REFITS INSTEAD OF REUSING A SAVED PLANE
----------------------------------------------
A plane saved from an earlier capture is only correct while the rig
sits exactly where it was. Move it and the saved wall is in the wrong
place -- which is what broke the earlier show_walls.py overlay. Refitting
live means the grid follows the rig. The plane is smoothed over time so
it does not jitter, and L freezes it once it looks right.


WHICH PLANE
-----------
The one under the image centre, not simply the largest. A textured desk
in the foreground has far more matched pixels than a blank wall, so
"largest" picks the desk. Aim the centre box at the wall; if the box
sits on featureless paint and finds no depth, aim it at a marker or
anything with texture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os as _os
_os.environ.setdefault("QT_LOGGING_RULES", "*=false")
_os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stereo_camera import CameraSettings, SyncedStereo          # noqa: E402


def fit_plane_at(P, centre, thresh, iters=500, rng=None):
    """RANSAC plane constrained to pass near `centre`."""
    rng = rng or np.random.default_rng(0)
    if len(P) < 60:
        return None
    best, bc = None, 0
    for _ in range(iters):
        p0, p1, p2 = P[rng.choice(len(P), 3, replace=False)]
        nr = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nr)
        if ln < 1e-9:
            continue
        nr /= ln
        d = float(-nr @ p0)
        if abs(centre @ nr + d) > thresh:       # must contain the centre
            continue
        c = int((np.abs(P @ nr + d) < thresh).sum())
        if c > bc:
            best, bc = (nr, d), c
    if best is None:
        return None
    nr, d = best
    inl = np.abs(P @ nr + d) < thresh
    c = P[inl].mean(0)
    _, _, Vt = np.linalg.svd(P[inl] - c, full_matrices=False)
    nr = Vt[2]
    if nr[2] > 0:                                # face the camera
        nr = -nr
    d = float(-nr @ c)
    return nr, d, int(inl.sum())


def project(X, P1):
    """mm in the left rectified frame -> pixels; None if behind camera."""
    X = np.atleast_2d(X)
    ok = X[:, 2] > 1.0
    uv = np.full((len(X), 2), np.nan)
    uv[ok, 0] = P1[0, 0] * X[ok, 0] / X[ok, 2] + P1[0, 2]
    uv[ok, 1] = P1[1, 1] * X[ok, 1] / X[ok, 2] + P1[1, 2]
    return uv, ok


def draw_line3d(img, a, b, P1, colour, thick, n=24):
    """Draw a straight 3D segment by sampling it, so it bends correctly
    only where the projection genuinely bends it."""
    pts = np.linspace(a, b, n)
    uv, ok = project(pts, P1)
    for i in range(n - 1):
        if ok[i] and ok[i + 1]:
            p = tuple(np.round(uv[i]).astype(int))
            q = tuple(np.round(uv[i + 1]).astype(int))
            cv2.line(img, p, q, colour, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="stereo_calib.npz")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--spacing-mm", type=float, default=100.0)
    ap.add_argument("--extent-mm", type=float, default=1200.0,
                    help="how far the grid reaches from the centre point")
    ap.add_argument("--num-disp", type=int, default=96)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--disp-offset", type=float, default=None,
                    help="defaults to disp_offset saved in the calibration")
    ap.add_argument("--refit-every", type=int, default=3)
    ap.add_argument("--history", type=int, default=15,
                    help="fits the median is taken over. Higher is steadier "
                         "but slower to follow when you move the rig")
    a = ap.parse_args()

    size = (a.width, a.height)
    with np.load(a.calib) as f:
        K1, D1, K2, D2 = f["K1"], f["D1"], f["K2"], f["D2"]
        R, T = f["R"], f["T"]
        saved_off = float(f["disp_offset"]) if "disp_offset" in f.files else 0.0
    off = saved_off if a.disp_offset is None else a.disp_offset

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    mL = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
    mR = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2)
    fx = float(P1[0, 0])
    b = abs(float(np.asarray(T).ravel()[0]))
    fB = fx * b
    print(f"fx {fx:.1f}  baseline {b:.2f} mm  disparity offset {off:.2f} px")

    # Identical to capture_wall.py -- the offset was validated with these.
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=a.num_disp, blockSize=a.block,
        P1=8 * a.block ** 2, P2=32 * a.block ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    spacing = a.spacing_mm
    plane = None            # smoothed (normal, d)
    # Recent fits. The plane shown is their MEDIAN, not a running average:
    # a single bad fit -- a frame where RANSAC grabbed the skirting board,
    # or the centre patch landed on a patch of noise -- moves a running
    # average and does nothing to a median. That is what stops the
    # distance and tilt flickering.
    from collections import deque
    hist = deque(maxlen=a.history)
    locked = False
    frame = 0
    rng = np.random.default_rng(0)
    win = "live grid"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("aim the centre box at the wall | +/- spacing | L lock | Q quit")

    with SyncedStereo(settings=CameraSettings(width=a.width,
                                              height=a.height)) as cam:
        while True:
            L, Rr = cam.capture()
            rL = cv2.remap(L, *mL, cv2.INTER_LINEAR)
            rR = cv2.remap(Rr, *mR, cv2.INTER_LINEAR)
            h, w = rL.shape[:2]
            frame += 1
            status = ""

            if not locked and frame % a.refit_every == 0:
                gL = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY)
                gR = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)
                disp = matcher.compute(gL, gR).astype(np.float32) / 16.0
                if off:
                    disp = np.where(disp > 0, disp - off, disp)
                valid = disp > 0.5
                xyz = cv2.reprojectImageTo3D(
                    np.where(valid, disp, 1.0).astype(np.float32), Q)

                cy, cx = h // 2, w // 2
                sl = (slice(cy - 20, cy + 20), slice(cx - 20, cx + 20))
                cpts = xyz[sl][valid[sl]]
                cpts = cpts[np.isfinite(cpts).all(1) & (cpts[:, 2] > 0)]
                if len(cpts) >= 15:
                    centre = np.median(cpts, 0)
                    P = xyz[valid]
                    P = P[np.isfinite(P).all(1) & (P[:, 2] > 0)
                          & (P[:, 2] < 6000)]
                    if len(P) > 8000:
                        P = P[rng.choice(len(P), 8000, replace=False)]
                    zc = float(centre[2])
                    thresh = max(3 * 0.25 * zc * zc / fB, 10.0)
                    fit = fit_plane_at(P, centre, thresh, rng=rng)
                    if fit is not None:
                        nr, d, _ = fit
                        if hist and hist[-1][0] @ nr < 0:
                            nr, d = -nr, -d          # keep a consistent sign
                        hist.append((nr, d))
                        N = np.array([h_[0] for h_ in hist])
                        D = np.array([h_[1] for h_ in hist])
                        n_med = np.median(N, axis=0)
                        n_med /= np.linalg.norm(n_med)
                        plane = (n_med, float(np.median(D)))
                else:
                    status = "no depth at centre -- aim at texture"

            shown = rL.copy()
            cv2.rectangle(shown, (w // 2 - 20, h // 2 - 20),
                          (w // 2 + 20, h // 2 + 20), (255, 255, 255), 1)

            if plane is not None:
                nr, d = plane
                # Anchor: where the optical axis meets the plane.
                ray = np.array([0.0, 0.0, 1.0])
                denom = nr @ ray
                if abs(denom) > 1e-6:
                    anchor = ray * (-d / denom)
                    # In-plane axes: horizontal first, then "up the wall".
                    horiz = np.cross([0.0, 1.0, 0.0], nr)
                    if np.linalg.norm(horiz) < 1e-6:
                        horiz = np.cross([1.0, 0.0, 0.0], nr)
                    horiz /= np.linalg.norm(horiz)
                    vert = np.cross(nr, horiz)
                    E = a.extent_mm
                    k = int(E // spacing)
                    for i in range(-k, k + 1):
                        major = (i % 5 == 0)
                        col = (0, 255, 255) if major else (0, 180, 0)
                        th = 2 if major else 1
                        o = anchor + horiz * (i * spacing)
                        draw_line3d(shown, o - vert * E, o + vert * E, P1, col, th)
                        o = anchor + vert * (i * spacing)
                        draw_line3d(shown, o - horiz * E, o + horiz * E, P1, col, th)
                    dist = abs(d)
                    tilt = np.degrees(np.arccos(min(1.0, abs(nr[2]))))
                    txt = [f"wall {round(dist / 5) * 5:6.0f} mm   "
                           f"tilt {tilt:4.1f} deg   "
                           f"({len(hist)}/{a.history} fits)"
                           + ("   LOCKED" if locked else ""),
                           f"grid {spacing:.0f} mm  (bright line every "
                           f"{5 * spacing:.0f} mm)   skew {cam.skew_ms*1000:.0f} us"]
                else:
                    txt = ["wall edge-on to the camera"]
            else:
                txt = ["looking for the wall..."]
            if status:
                txt.append(status)

            for i, t in enumerate(txt):
                for thk, c in ((3, (0, 0, 0)), (1, (0, 255, 0))):
                    cv2.putText(shown, t, (8, 22 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, thk,
                                cv2.LINE_AA)
            cv2.imshow(win, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("+"), ord("=")):
                spacing = min(spacing * 2, 1000.0)
            elif key in (ord("-"), ord("_")):
                spacing = max(spacing / 2, 25.0)
            elif key == ord("l"):
                locked = not locked
            elif key == ord("r"):
                hist.clear()
                plane = None
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()