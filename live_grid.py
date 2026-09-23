#!/usr/bin/env python3
"""
Live, distance-accurate gridlines drawn onto the wall in the camera view.

    python3 live_grid.py
    python3 live_grid.py --spacing-mm 100 --extent-mm 1500

    +/-  change grid spacing
    L    lock / unlock the plane (freeze it once it looks right)
    R    reset -- use after moving the rig
    T    show / hide which pixels are trusted
    S    save the current rectified left/right pair to captures/ (raw,
         before the row filter) -- for diagnosing the corruption
    Q    quit

Hold the rig still facing the wall. The last --window frames are combined
the same way capture_wall.py combines a burst, the wall plane is fitted to
what survives, and a square grid is laid on it and drawn into the image.
The grid lives on the measured plane in millimetres, so each line really
is --spacing-mm from the next.


THE ROW-LINE FILTER
-------------------
The extenders corrupt whole single pixel rows, and on this rig the
corruption persists long enough to survive burst averaging -- measured: a
30-frame capture's steady points formed steep phantom planes at 60-70 deg
and the wall was not even a candidate. So the lines are removed BEFORE
matching, with a 3-row vertical median on each image. A one-row feature
cannot survive it; real texture spans many rows and passes through. It
acts vertically, disparity is horizontal, so real-texture disparities are
not shifted and the validated offset still applies. --row-filter 5 handles
lines two rows thick; --row-filter 0 turns it off for comparison.


WHY A ROLLING WINDOW, NOT SINGLE FRAMES
---------------------------------------
The CSI-to-HDMI extenders add horizontal-line corruption that changes from
frame to frame, independently in each camera. On blank paint SGBM turns
those lines into false matches, and because each line is a row, the false
depths line up into a steep phantom plane. A single frame cannot tell that
phantom from a real wall, and no image-gradient test separates them.

Time does. The noise moves; the wall's real features -- the ceiling line,
the pillar corner, the red square -- stay where they are. So a pixel is
trusted only if it was valid in at least --min-valid-frac of the window
and its disparity barely moved (std <= --max-std px), and its disparity is
the mean over the window. These are exactly capture_wall.py's rules and
defaults, the ones the tape-validated results came from.


WHY THE MATCHER SETTINGS MATCH capture_wall.py EXACTLY
------------------------------------------------------
Same SGBM parameters, NO WLS filter, the same disparity offset, applied per
frame before averaging as capture_wall.py does. The 1.15 px offset was
validated against a tape at 1.7 m and 2.5 m with exactly these settings.


WHICH PLANE
-----------
With trusted depth under the centre box, the plane is forced through that
point -- not simply the largest plane, which a textured desk in the
foreground would otherwise win.

With blank paint under the centre box -- or when the plane through the
centre is rejected, because the centre depth is itself a steady false
match -- the wall is found from its edges:
planes are peeled off the trusted points, and the one chosen is the one
whose points SURROUND the centre in the image. A blank wall's edges enclose
its blank middle; the pillar, a doorway or floor clutter sit off to one
side. Only contiguous patches count -- isolated false matches scattered
over the paint must not stretch an unrelated plane around the centre. A candidate is rejected if its points lie along a single edge (the
plane could pivot about it), or if its separate patches face different
ways (it is joining two unrelated objects).

Every candidate must also be farther than the rig can resolve and tilted
less than --max-tilt. Rejections are shown on screen with the reason.

When a refit fails, the wall already on screen is tested against the
current steady points. If it still has at least --hold-frac of the support
it had when it was accepted, it is kept: on a
blank wall the steady depth is thin, so individual refits miss even though
the wall has not moved. Only when the data stops supporting it for
--stale-after refits in a row is it dropped. Press R after moving the rig.
The plane drawn is the median of the last --history accepted fits.


WHAT THE READOUT MEANS
----------------------
axis     distance along the camera's viewing axis to the wall -- the one
         to compare with a tape run straight out from the lens.
normal   perpendicular distance from the lens to the wall plane.
tilt     angle between the wall's normal and the viewing axis. 0 = square on.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections import deque
from pathlib import Path

os.environ.setdefault("QT_LOGGING_RULES", "*=false")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------- stereo

def load_rectification(calib, size):
    with np.load(calib) as f:
        K1, D1, K2, D2 = f["K1"], f["D1"], f["K2"], f["D2"]
        R, T = f["R"], f["T"]
        off = float(f["disp_offset"]) if "disp_offset" in f.files else 0.0
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return dict(
        mL=cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2),
        mR=cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2),
        P1=P1, Q=Q, fx=float(P1[0, 0]),
        baseline=abs(float(np.asarray(T).ravel()[0])), offset=off)


def make_matcher(num_disp, block):
    """Identical to capture_wall.py -- the offset was validated with these."""
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * block ** 2, P2=32 * block ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def remove_row_lines(gray, size=3):
    """Vertical median over `size` rows. The CSI-to-HDMI extenders corrupt
    thin runs of pixel rows; any feature thinner than size/2 rows is
    removed, while real texture -- which spans many rows -- passes through.
    Filtering is vertical only and disparity is horizontal, so disparities
    on real texture are unchanged."""
    if size < 3:
        return gray
    if size == 3:                                   # fast path
        a, b, c = gray[:-2], gray[1:-1], gray[2:]
        mid = np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))
        out = gray.copy()
        out[1:-1] = mid
        return out
    k = size // 2
    pad = np.pad(gray, ((k, k), (0, 0)), mode="edge")
    stack = np.stack([pad[i:i + gray.shape[0]] for i in range(size)])
    return np.median(stack, axis=0).astype(gray.dtype)


def consistent_disparity(stack, min_valid_frac, max_std):
    """capture_wall.py's burst rule. Returns (mean disparity, trusted mask)."""
    D = np.stack(stack)
    valid = D > 0
    frac = valid.mean(axis=0)
    Dm = np.where(valid, D, np.nan)
    with warnings.catch_warnings():                  # all-NaN pixels are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nan_to_num(np.nanmean(Dm, axis=0), nan=0.0)
        std = np.nan_to_num(np.nanstd(Dm, axis=0), nan=1e9)
    trusted = (frac >= min_valid_frac) & (std <= max_std) & (mean > 0.5)
    return mean.astype(np.float32), trusted


# ------------------------------------------------------------- planes

def tilt_deg(nr):
    return float(np.degrees(np.arccos(min(1.0, abs(nr[2])))))


def _refit(P, inl):
    c = P[inl].mean(0)
    nr = np.linalg.svd(P[inl] - c, full_matrices=False)[2][2]
    if nr[2] > 0:                                    # face the camera
        nr = -nr
    return nr, float(-nr @ c)


def ransac_plane(P, thresh, centre=None, iters=400, rng=None):
    """RANSAC + SVD refit. With `centre`, only planes through it count.
    Returns (normal, d, inlier mask) or None."""
    rng = rng or np.random.default_rng(0)
    if len(P) < 60:
        return None
    best, best_count = None, 0
    for _ in range(iters):
        p0, p1, p2 = P[rng.choice(len(P), 3, replace=False)]
        nr = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nr)
        if ln < 1e-9:
            continue
        nr /= ln
        d = float(-nr @ p0)
        if centre is not None and abs(centre @ nr + d) > thresh:
            continue
        count = int((np.abs(P @ nr + d) < thresh).sum())
        if count > best_count:
            best, best_count = (nr, d), count
    if best is None:
        return None
    nr, d = best
    nr, d = _refit(P, np.abs(P @ nr + d) < thresh)
    return nr, d, np.abs(P @ nr + d) < thresh


def spread_ok(pts, nr, ratio=0.15):
    """False if the points lie nearly along a line on the plane."""
    u = np.cross(nr, [0.0, 1.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(nr, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(nr, u)
    su, sv = np.ptp(pts @ u), np.ptp(pts @ v)
    return min(su, sv) >= ratio * max(su, sv)


def patches_agree(xyz, trusted, nr, d, thresh, max_dev=12.0, min_pts=150):
    """Fit each broad connected patch of the plane's inliers on its own. A
    real wall's patches all share its orientation; a plane joining two
    unrelated objects does not. Thin strips are skipped. -> (ok, worst deg)"""
    dist = np.abs(np.nan_to_num(xyz @ nr + d, nan=1e9, posinf=1e9, neginf=1e9))
    inl = (trusted & (dist < thresh)).astype(np.uint8)
    n_lab, lab = cv2.connectedComponents(cv2.dilate(inl, np.ones((5, 5), np.uint8)))
    worst = 0.0
    for k in range(1, n_lab):
        pts = xyz[(lab == k) & (inl > 0)]
        if len(pts) < min_pts or not spread_ok(pts, nr, ratio=0.3):
            continue
        c = pts.mean(0)
        pn = np.linalg.svd(pts - c, full_matrices=False)[2][2]
        worst = max(worst, float(np.degrees(np.arccos(min(1.0, abs(pn @ nr))))))
    return worst <= max_dev, worst


def surrounds_centre(xyz, trusted, nr, d, thresh, centre_px, min_patch=40):
    """True if the plane's contiguous patches enclose the image centre.
    Isolated pixels -- steady false matches scattered over blank paint --
    are ignored: a few of them at the right depth can otherwise stretch
    the outline of an unrelated plane around the centre."""
    dist = np.abs(np.nan_to_num(xyz @ nr + d, nan=1e9, posinf=1e9, neginf=1e9))
    inl = (trusted & (dist < thresh)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inl, connectivity=8)
    big = np.nonzero(stats[1:, cv2.CC_STAT_AREA] >= min_patch)[0] + 1
    if not len(big):
        return False
    ys, xs = np.nonzero(np.isin(lab, big))
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    return cv2.pointPolygonTest(hull, centre_px, False) > 0


def fit_frame(disp, trusted, Q, fB, num_disp, max_tilt, min_inliers, rng=None,
              max_planes=8, log=None):
    """Fit the wall to consistent disparity. Returns ((normal, d), note) or
    (None, reason). disp already has the offset applied and is averaged.
    Pass a list as `log` to receive one line per candidate plane."""
    def note_(msg):
        if log is not None:
            log.append(msg)
    rng = rng or np.random.default_rng(0)
    h, w = disp.shape
    z_min = fB / num_disp
    xyz = cv2.reprojectImageTo3D(np.where(trusted, disp, 1.0).astype(np.float32), Q)

    ys, xs = np.nonzero(trusted)
    P = xyz[ys, xs]
    keep = np.isfinite(P).all(1) & (P[:, 2] > z_min) & (P[:, 2] < 6000)
    P, ys, xs = P[keep], ys[keep], xs[keep]
    if len(P) < min_inliers:
        return None, "too little steady depth -- hold still, get wall edges in view"
    if len(P) > 8000:
        sel = rng.choice(len(P), 8000, replace=False)
        P, ys, xs = P[sel], ys[sel], xs[sel]

    in_box = (np.abs(ys - h // 2) < 20) & (np.abs(xs - w // 2) < 20)
    centre_why = None
    if in_box.sum() >= 15:
        # --- steady depth under the centre: plane through that point
        centre = np.median(P[in_box], 0)
        zc = float(centre[2])
        thresh = max(3 * 0.25 * zc * zc / fB, 10.0)
        r = ransac_plane(P, thresh, centre=centre, rng=rng)
        if r is None:
            centre_why = "no plane through the centre"
        else:
            nr, d, inl = r
            if tilt_deg(nr) > max_tilt:
                centre_why = f"tilt {tilt_deg(nr):.0f} deg > {max_tilt:.0f}"
            elif inl.sum() < min_inliers:
                centre_why = f"only {int(inl.sum())} points on the plane"
            else:
                note_(f"centre ({zc:.0f} mm deep): plane accepted")
                return (nr, d), ""
        note_(f"centre ({zc:.0f} mm deep): {centre_why} -> trying the edges")
    else:
        note_(f"centre: only {int(in_box.sum())} steady points -> trying the edges")
        # The centre depth itself may be a steady false match -- noise that
        # survived the window, often near the closest measurable range.
        # Fall through and find the wall from its edges instead.

    # --- blank (or untrustworthy) centre: the plane whose points surround it
    ray = np.array([(w // 2 + Q[0, 3]) / Q[2, 3], (h // 2 + Q[1, 3]) / Q[2, 3], 1.0])
    rest = np.arange(len(P))
    best, best_t, why = None, np.inf, "no plane surrounds the centre"
    for k in range(max_planes):
        if len(rest) < min_inliers:
            note_(f"stopped: {len(rest)} points left")
            break
        zmed = float(np.median(P[rest, 2]))
        thresh = max(3 * 0.25 * zmed * zmed / fB, 10.0)
        r = ransac_plane(P[rest], thresh, rng=rng)
        if r is None:
            break
        nr, d, inl = r
        idx, rest = rest[inl], rest[~inl]
        denom = nr @ ray
        axis = -d / denom if abs(denom) > 1e-6 else float("inf")
        tag = (f"plane {k + 1}: axis {axis:7.0f} mm  tilt {tilt_deg(nr):4.1f} deg  "
               f"{len(idx):5d} pts  ")
        if len(idx) < min_inliers:
            note_(tag + "too few points")
            continue
        if not surrounds_centre(xyz, trusted, nr, d, thresh,
                                (float(w // 2), float(h // 2))):
            note_(tag + "does not surround the centre")
            continue
        if abs(denom) < 1e-6 or axis < z_min:
            note_(tag + "closer than measurable")
            continue
        if tilt_deg(nr) > max_tilt:
            why = f"rejected: tilt {tilt_deg(nr):.0f} deg > {max_tilt:.0f}"
            note_(tag + why)
            continue
        if not spread_ok(P[idx], nr):
            why = "rejected: wall points lie along one edge only"
            note_(tag + why)
            continue
        agree, dev = patches_agree(xyz, trusted, nr, d, thresh)
        if not agree:
            why = f"rejected: plane joins separate surfaces ({dev:.0f} deg apart)"
            note_(tag + why)
            continue
        note_(tag + "CANDIDATE")
        if axis < best_t:                                # front-most wins
            best, best_t = (nr, d), axis
    if best is None:
        return None, (f"centre fit rejected ({centre_why}); edges: {why}"
                      if centre_why else why)
    return best, ("centre depth rejected -- wall found from its edges"
                  if centre_why else "blank centre -- wall found from its edges")


def plane_support(disp, trusted, Q, fB, plane, z_min):
    """Steady points lying on `plane`, at the tolerance for its distance.
    Used to keep a wall that still fits the data when a refit fails."""
    nr, d = plane
    if abs(nr[2]) < 1e-6:
        return 0
    xyz = cv2.reprojectImageTo3D(np.where(trusted, disp, 1.0).astype(np.float32), Q)
    P = xyz[trusted]
    P = P[np.isfinite(P).all(1) & (P[:, 2] > z_min) & (P[:, 2] < 6000)]
    z = -d / nr[2]
    thresh = max(3 * 0.25 * z * z / fB, 10.0)
    return int((np.abs(P @ nr + d) < thresh).sum())


def median_plane(hist):
    n = np.median(np.array([p[0] for p in hist]), axis=0)
    n /= np.linalg.norm(n)
    return n, float(np.median([p[1] for p in hist]))


# ------------------------------------------------------------ drawing

def project(X, P1):
    X = np.atleast_2d(X)
    ok = X[:, 2] > 1.0
    uv = np.full((len(X), 2), np.nan)
    uv[ok, 0] = P1[0, 0] * X[ok, 0] / X[ok, 2] + P1[0, 2]
    uv[ok, 1] = P1[1, 1] * X[ok, 1] / X[ok, 2] + P1[1, 2]
    return uv, ok


def draw_line3d(img, a, b, P1, colour, thick, n=24):
    uv, ok = project(np.linspace(a, b, n), P1)
    for i in range(n - 1):
        if ok[i] and ok[i + 1]:
            cv2.line(img, tuple(np.round(uv[i]).astype(int)),
                     tuple(np.round(uv[i + 1]).astype(int)), colour, thick, cv2.LINE_AA)


def draw_grid(img, plane, P1, spacing, extent):
    """Grid anchored where the viewing axis meets the plane. -> anchor or None."""
    nr, d = plane
    if abs(nr[2]) < 1e-6:
        return None
    anchor = np.array([0.0, 0.0, -d / nr[2]])
    horiz = np.cross([0.0, 1.0, 0.0], nr)
    if np.linalg.norm(horiz) < 1e-6:
        horiz = np.cross([1.0, 0.0, 0.0], nr)
    horiz /= np.linalg.norm(horiz)
    vert = np.cross(nr, horiz)
    k = int(extent // spacing)
    for i in range(-k, k + 1):
        col, th = ((0, 255, 255), 2) if i % 5 == 0 else ((0, 180, 0), 1)
        o = anchor + horiz * (i * spacing)
        draw_line3d(img, o - vert * extent, o + vert * extent, P1, col, th)
        o = anchor + vert * (i * spacing)
        draw_line3d(img, o - horiz * extent, o + horiz * extent, P1, col, th)
    return anchor


def put_lines(img, lines):
    for i, t in enumerate(lines):
        for thk, c in ((3, (0, 0, 0)), (1, (0, 255, 0))):
            cv2.putText(img, t, (8, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, c, thk, cv2.LINE_AA)


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default="stereo_calib.npz")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--spacing-mm", type=float, default=100.0)
    ap.add_argument("--extent-mm", type=float, default=1200.0)
    ap.add_argument("--num-disp", type=int, default=96)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--disp-offset", type=float, default=None,
                    help="defaults to disp_offset saved in the calibration")
    ap.add_argument("--row-filter", type=int, default=3,
                    help="rows in the vertical median that removes the HDMI "
                         "line corruption; 3 removes 1-row lines, 5 removes "
                         "up to 2-row lines, 0 turns it off for comparison")
    ap.add_argument("--window", type=int, default=12,
                    help="frames combined, as capture_wall.py combines a burst")
    ap.add_argument("--min-valid-frac", type=float, default=0.6)
    ap.add_argument("--max-std", type=float, default=1.0)
    ap.add_argument("--max-tilt", type=float, default=55.0)
    ap.add_argument("--min-inliers", type=int, default=300)
    ap.add_argument("--refit-every", type=int, default=3)
    ap.add_argument("--history", type=int, default=15)
    ap.add_argument("--stale-after", type=int, default=10)
    ap.add_argument("--hold-frac", type=float, default=0.5,
                    help="keep the wall on a failed refit while it still has "
                         "this fraction of the support it had when accepted")
    a = ap.parse_args()

    from stereo_camera import CameraSettings, SyncedStereo

    rect = load_rectification(a.calib, (a.width, a.height))
    off = rect["offset"] if a.disp_offset is None else a.disp_offset
    fB = rect["fx"] * rect["baseline"]
    print(f"fx {rect['fx']:.1f}  baseline {rect['baseline']:.2f} mm  "
          f"offset {off:.2f} px  nearest measurable {fB / a.num_disp:.0f} mm")
    matcher = make_matcher(a.num_disp, a.block)

    spacing = a.spacing_mm
    stack = deque(maxlen=a.window)
    hist = deque(maxlen=a.history)
    plane, trusted, ref_support = None, None, 0
    locked = show_trust = False
    rejects, frame, status = 0, 0, ""
    rng = np.random.default_rng(0)
    win = "live grid"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("hold still facing the wall | +/- spacing | L lock | R reset "
          "| T trusted | S save pair | Q quit")

    with SyncedStereo(settings=CameraSettings(width=a.width,
                                              height=a.height)) as cam:
        while True:
            L, Rr = cam.capture()
            rL = cv2.remap(L, *rect["mL"], cv2.INTER_LINEAR)
            rR = cv2.remap(Rr, *rect["mR"], cv2.INTER_LINEAR)
            h, w = rL.shape[:2]
            frame += 1

            if not locked:
                gL = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY)
                gR = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)
                if a.row_filter:
                    gL = remove_row_lines(gL, a.row_filter)
                    gR = remove_row_lines(gR, a.row_filter)
                disp = matcher.compute(gL, gR).astype(np.float32) / 16.0
                if off:
                    disp = np.where(disp > 0, disp - off, disp)
                stack.append(disp)

                if len(stack) == a.window and frame % a.refit_every == 0:
                    mean, trusted = consistent_disparity(
                        stack, a.min_valid_frac, a.max_std)
                    fit, status = fit_frame(mean, trusted, rect["Q"], fB,
                                            a.num_disp, a.max_tilt,
                                            a.min_inliers, rng)
                    if fit is not None:
                        nr, d = fit
                        if hist and hist[-1][0] @ nr < 0:
                            nr, d = -nr, -d
                        hist.append((nr, d))
                        plane = median_plane(hist)
                        ref_support = plane_support(mean, trusted, rect["Q"], fB,
                                                    plane, fB / a.num_disp)
                        rejects = 0
                    elif plane is not None and plane_support(
                            mean, trusted, rect["Q"], fB, plane, fB / a.num_disp
                            ) >= max(a.min_inliers, a.hold_frac * ref_support):
                        rejects = 0                  # wall still fits the data
                        status = "holding: refit missed, current wall still fits"
                    else:
                        rejects += 1
                        if rejects >= a.stale_after:
                            hist.clear()
                            plane = None

            shown = rL.copy()
            if show_trust and trusted is not None:
                shown[trusted] = (0.6 * shown[trusted] + (0, 100, 0)).astype(np.uint8)
            cv2.rectangle(shown, (w // 2 - 20, h // 2 - 20),
                          (w // 2 + 20, h // 2 + 20), (255, 255, 255), 1)

            txt = []
            if len(stack) < a.window and not locked:
                txt.append(f"gathering frames {len(stack)}/{a.window} -- hold still")
            elif plane is not None:
                anchor = draw_grid(shown, plane, rect["P1"], spacing, a.extent_mm)
                if anchor is None:
                    txt.append("wall edge-on to the camera")
                else:
                    nr, d = plane
                    txt.append(f"axis {anchor[2]:5.0f} mm   normal {abs(d):5.0f} mm   "
                               f"tilt {tilt_deg(nr):4.1f} deg   ({len(hist)}/{a.history})"
                               + ("   LOCKED" if locked else ""))
                    txt.append(f"grid {spacing:.0f} mm (bright every {5 * spacing:.0f})"
                               f"   skew {cam.skew_ms * 1000:.0f} us")
            else:
                txt.append("looking for the wall...")
            if status:
                txt.append(status + (f"  [{rejects} rejected]" if rejects else ""))
            put_lines(shown, txt)
            cv2.imshow(win, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("+"), ord("=")):
                spacing = min(spacing * 2, 1000.0)
            elif key in (ord("-"), ord("_")):
                spacing = max(spacing / 2, 25.0)
            elif key == ord("l"):
                locked = not locked
            elif key == ord("r"):
                stack.clear()
                hist.clear()
                plane, trusted, rejects, status = None, None, 0, ""
                ref_support = 0
            elif key == ord("t"):
                show_trust = not show_trust
            elif key == ord("s"):
                out = Path("captures")
                out.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(out / f"pair_{stamp}_L.png"), rL)
                cv2.imwrite(str(out / f"pair_{stamp}_R.png"), rR)
                status = f"saved captures/pair_{stamp}_L.png and _R.png"
                print(status)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()