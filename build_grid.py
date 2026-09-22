#!/usr/bin/env python3
"""
Build a 3D occupancy grid of the room by sweeping the stereo head.

    python3 -m tests.build_grid
    python3 -m tests.build_grid --voxel-mm 20 --extent-mm 3000
    python3 -m tests.build_grid --no-wls          # faster, holier

    SPACE  start / pause accumulating
    R      reset the grid and the angle back to zero
    D      cycle view: live | disparity | top-down grid
    S      save grid.npz and cloud.ply
    Q      quit


WHAT THIS DOES
--------------
The instant you start, wherever the head happens to be pointing becomes
the world. Origin sits midway between the two lenses; +X right, +Y down,
+Z forward, which is OpenCV's convention so nothing needs converting.

THAT FRAME THEN STOPS MOVING. Pan the head and the cameras travel through
it. Every frame's point cloud is measured from wherever the head is now,
so it gets rotated back by the running angle estimate before being
dropped into the grid. A chair 1.2 m ahead stays at the same coordinates
whether it was seen at -40 degrees or +10.

Space is then cut into cubes of --voxel-mm. A cube that any point lands
in is occupied. Sweep, and the cubes fill in across the whole arc.


THE ANGLE, AND WHY THIS IS THE WEAK PART
----------------------------------------
With no encoder the rotation has to come from the images. Consecutive
frames overlap heavily and the axis is known to be vertical, so this is
a ONE unknown problem rather than the six of general SLAM: a feature
that shifts by du pixels between two frames implies

    dtheta ~ du / fx

ORB features, matched frame to frame, median shift, accumulated. That
much is sound.

What is not sound is doing it by hand. Two separate errors:

  THE PIVOT WANDERS. A motor turns about a fixed axis. A hand does not,
  so every frame carries an unknown translation as well as a rotation.
  This code assumes pure rotation and has no way to see the difference,
  so hand translation is silently absorbed as extra angle -- which is
  wrong, and the grid smears by however far your hand moved.

  DRIFT ACCUMULATES. Each estimate carries perhaps 0.2 degrees of error
  and they simply add. Over 180 degrees in 40 frames that is a few
  degrees by the far end, and at 2 m a 2 degree error is 70 mm -- more
  than three voxels. The far end of the sweep will be visibly worse than
  the near end.

Rest the rig on something that pivots about a fixed vertical axis and the
first error mostly goes away. An encoder removes both.

Watch the top-down view while sweeping: a straight wall is the test. If
it comes out straight the geometry is holding; if it curves, that curve
is the drift.


RANGE, AND WHY THE GRID IS NOT HONEST EVERYWHERE
------------------------------------------------
Depth error grows with the SQUARE of distance. At fB = 54000 a quarter
pixel of disparity error is

     1 m   +/- 4.6 mm      within one 20 mm voxel
     2 m   +/- 19 mm       about one voxel
     3 m   +/- 42 mm       two voxels
     5 m   +/- 116 mm      six voxels

So 20 mm cells are meaningful out to roughly 2 m and increasingly
decorative beyond. --max-depth-mm defaults to 2500 for that reason. Raise
it if you want the walls, but do not trust cell boundaries out there.
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

from stereo_camera import CameraSettings, SyncedStereo         # noqa: E402


# ==========================================================================
class Rectifier:
    """
    Rectification straight from the calibration file.

    Inlined rather than imported so this package stands alone. It is the
    standard stereoRectify + initUndistortRectifyMap pair, with alpha=0
    so the output contains only pixels valid in both views -- no black
    wedges at the borders to confuse the matcher into finding structure
    that is not there.
    """

    def __init__(self, npz_path, size):
        with np.load(npz_path) as f:
            self.K1, self.D1 = f["K1"], f["D1"]
            self.K2, self.D2 = f["K2"], f["D2"]
            self.R, self.T = f["R"], f["T"]

        tx = float(np.asarray(self.T).ravel()[0])
        if tx > 0:
            print(f"[calib] WARNING Tx = {tx:+.2f} mm is POSITIVE. The two "
                  f"eyes are swapped, every reconstructed point lands "
                  f"behind the cameras, and the grid will be empty while "
                  f"the preview looks perfectly normal.")
        self.baseline = abs(tx)

        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.m1 = cv2.initUndistortRectifyMap(
            self.K1, self.D1, R1, P1, size, cv2.CV_16SC2)
        self.m2 = cv2.initUndistortRectifyMap(
            self.K2, self.D2, R2, P2, size, cv2.CV_16SC2)
        self.fx = float(P1[0, 0])
        self.fB = self.fx * self.baseline

    def rectify(self, left, right):
        return (cv2.remap(left, self.m1[0], self.m1[1], cv2.INTER_LINEAR),
                cv2.remap(right, self.m2[0], self.m2[1], cv2.INTER_LINEAR))

    def describe(self):
        return (f"fx {self.fx:.1f} px  baseline {self.baseline:.2f} mm  "
                f"fB {self.fB:.0f}")


# ==========================================================================
def build_sgbm(num_disparities, block_size, want_wls=True):
    nd = int(np.ceil(num_disparities / 16.0) * 16)
    left = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=block_size,
        P1=8 * 3 * block_size ** 2, P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=150, speckleRange=2, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    if not want_wls:
        return left, None, None, nd
    try:
        right = cv2.ximgproc.createRightMatcher(left)
        wls = cv2.ximgproc.createDisparityWLSFilter(left)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
        return left, right, wls, nd
    except (AttributeError, cv2.error):
        print("[sgbm] no cv2.ximgproc -- running unfiltered")
        return left, None, None, nd


# ==========================================================================
class PoseTracker:
    """
    Full 6-DOF pose from stereo, frame to frame.

    WHY THIS REPLACED THE YAW TRACKER
    ---------------------------------
    The previous version estimated ONE number -- yaw -- from horizontal
    image shift, then assumed the motion was a pure rotation about the
    camera's own Y axis. Three assumptions, all of them breakable:

      * that the motion is rotation only. It is not: a hand, or a rig
        offset from its pivot, translates as well, and translation was
        being silently absorbed as extra angle.
      * that the axis is the camera's Y. It is not if the rig is pitched,
        and then even a perfectly correct angle sweeps a cone, so a flat
        wall comes out curved.
      * that the estimate can be accumulated. Small per-frame bias adds
        up without bound; on a stationary scene the reported angle
        climbed on its own.

    The symptom of all three is the same and it is distinctive: a clean
    ARC, not a fuzzy band. Random error blurs a wall. Systematic error
    rotates every copy of it by a consistent amount about the origin, and
    a rotation about the origin moves points along a circle -- so the
    stack of copies traces one.

    There is no need to estimate any of it indirectly, because a stereo
    pair measures 3D points directly. Match features between two frames,
    look up where each one is in 3D, and solve for the rigid transform
    that maps one set onto the other. That is Kabsch: closed form, one
    SVD, no iteration and no initial guess. RANSAC around it rejects the
    mismatches.

    This recovers rotation AND translation about ANY axis, so all three
    assumptions above simply stop existing.

    WHAT IT DOES NOT FIX
    --------------------
    It is still relative: each frame is measured against the last, so
    error accumulates over a long sequence. That is ordinary visual
    odometry drift, it grows as the square root of the number of frames
    rather than linearly, and it is bounded in practice by keyframing.
    An encoder on the pan axis removes even that, and this code is
    arranged so the encoder can be dropped in as an override.
    """

    def __init__(self, Q, max_features=900, min_matches=20,
                 ransac_mm=60.0, ransac_iters=120, commit_mm=25.0):
        self.Q = Q
        self.orb = cv2.ORB_create(max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches
        self.ransac_mm = ransac_mm
        self.ransac_iters = ransac_iters
        # Below this the frame has not moved enough to measure reliably;
        # holding the keyframe rather than committing noise is what stops
        # the drift-while-stationary the yaw tracker suffered from.
        self.commit_mm = commit_mm

        self.pose = np.eye(4)          # world <- current camera
        self.kp = None
        self.des = None
        self.kxyz = None               # 3D of keyframe features
        self.last_ok = False
        self.last_n = 0
        self.last_inliers = 0
        self.commits = 0

    # ------------------------------------------------------------- utils
    @staticmethod
    def _kabsch(A, B):
        """Rigid transform taking A onto B. Closed form, no iteration."""
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        # The determinant guard matters: without it the SVD can return a
        # REFLECTION that fits the points beautifully and is physically
        # impossible, which shows up later as a mirrored scene.
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        return R, cb - R @ ca

    def _feature_xyz(self, kp, disp, valid):
        """3D position of each keypoint, or None where disparity is bad."""
        h, w = disp.shape
        uv = np.array([k.pt for k in kp])
        u = np.clip(uv[:, 0].astype(int), 0, w - 1)
        v = np.clip(uv[:, 1].astype(int), 0, h - 1)
        ok = valid[v, u]
        pts = cv2.perspectiveTransform(
            np.stack([uv[:, 0], uv[:, 1], disp[v, u]], 1
                     ).reshape(-1, 1, 3).astype(np.float32),
            self.Q).reshape(-1, 3)
        ok &= np.isfinite(pts).all(1) & (pts[:, 2] > 200) & (pts[:, 2] < 6000)
        return pts, ok

    # ------------------------------------------------------------ update
    def update(self, gray, disp, valid):
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(des) < 12:
            self.last_ok = False
            return self.pose
        xyz, ok = self._feature_xyz(kp, disp, valid)

        if self.des is None:
            self.kp, self.des, self.kxyz, self.kok = kp, des, xyz, ok
            self.last_ok = False
            return self.pose

        matches = self.bf.match(self.des, des)
        self.last_n = len(matches)
        if len(matches) < self.min_matches:
            self.kp, self.des, self.kxyz, self.kok = kp, des, xyz, ok
            self.last_ok = False
            return self.pose

        qi = np.array([m.queryIdx for m in matches])
        ti = np.array([m.trainIdx for m in matches])
        good = self.kok[qi] & ok[ti]
        A = xyz[ti][good]              # current frame
        B = self.kxyz[qi][good]        # keyframe
        if len(A) < self.min_matches:
            self.last_ok = False
            return self.pose

        # RANSAC: three points define a rigid transform, and a handful of
        # bad correspondences would otherwise dominate a least-squares fit.
        rng = np.random.default_rng(0)
        best_R, best_t, best_n = None, None, 0
        for _ in range(self.ransac_iters):
            idx = rng.choice(len(A), 3, replace=False)
            try:
                R, t = self._kabsch(A[idx], B[idx])
            except np.linalg.LinAlgError:
                continue
            err = np.linalg.norm((A @ R.T + t) - B, axis=1)
            n = int((err < self.ransac_mm).sum())
            if n > best_n:
                best_R, best_t, best_n = R, t, n
        if best_R is None or best_n < self.min_matches:
            self.last_ok = False
            return self.pose

        # Refit on all inliers -- RANSAC picks the right POINTS, Kabsch
        # then gets the best transform through them.
        err = np.linalg.norm((A @ best_R.T + best_t) - B, axis=1)
        inl = err < self.ransac_mm
        R, t = self._kabsch(A[inl], B[inl])
        self.last_inliers = int(inl.sum())
        self.last_ok = True

        if np.linalg.norm(t) < self.commit_mm and \
                np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2,
                                             -1, 1))) < 1.0:
            return self.pose        # not moved enough to be sure

        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        self.pose = self.pose @ T
        self.commits += 1
        self.kp, self.des, self.kxyz, self.kok = kp, des, xyz, ok
        return self.pose

    @property
    def yaw_deg(self):
        R = self.pose[:3, :3]
        return float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))

    @property
    def pitch_deg(self):
        R = self.pose[:3, :3]
        return float(np.degrees(np.arcsin(-np.clip(R[1, 2], -1, 1))))

    @property
    def translation_mm(self):
        return float(np.linalg.norm(self.pose[:3, 3]))

    def reset(self):
        self.pose = np.eye(4)
        self.kp = self.des = self.kxyz = None
        self.commits = 0


class VoxelGrid:
    """
    Occupancy on a regular lattice, centred on the origin.

    Stored as a count per cell rather than a bool. A cell seen once may be
    a stray disparity; a cell seen in twenty frames from several angles is
    really something. --min-hits then separates them at display time,
    which is far more useful than committing to a threshold up front.

    uint16 rather than uint8: a surface stared at for a few seconds
    saturates 255 easily, and once saturated you lose the ability to tell
    a solid wall from a speckle that got lucky.
    """

    def __init__(self, voxel_mm, extent_mm, height_mm):
        self.v = float(voxel_mm)
        self.nx = int(2 * extent_mm / voxel_mm)
        self.nz = int(2 * extent_mm / voxel_mm)
        self.ny = int(2 * height_mm / voxel_mm)
        self.extent = float(extent_mm)
        self.height = float(height_mm)
        self.grid = np.zeros((self.nx, self.ny, self.nz), np.uint16)
        print(f"[grid] {self.nx} x {self.ny} x {self.nz} cells of "
              f"{voxel_mm:.0f} mm = {self.grid.nbytes / 1e6:.0f} MB")

    def add(self, pts):
        """pts: Nx3 mm in the world frame."""
        if not len(pts):
            return 0
        ix = ((pts[:, 0] + self.extent) / self.v).astype(np.int32)
        iy = ((pts[:, 1] + self.height) / self.v).astype(np.int32)
        iz = ((pts[:, 2] + self.extent) / self.v).astype(np.int32)
        ok = ((ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
              & (iz >= 0) & (iz < self.nz))
        ix, iy, iz = ix[ok], iy[ok], iz[ok]
        # np.add.at is slow; this is the same thing via flat bincount.
        flat = (ix * self.ny + iy) * self.nz + iz
        counts = np.bincount(flat, minlength=self.grid.size)
        np.add(self.grid.reshape(-1), np.minimum(counts, 60).astype(np.uint16),
               out=self.grid.reshape(-1), casting="unsafe")
        return int(ok.sum())

    def occupied(self, min_hits):
        return int((self.grid >= min_hits).sum())

    def topdown(self, min_hits, size=420):
        """Bird's eye: max count down each vertical column."""
        col = self.grid.max(axis=1)                      # nx by nz
        img = np.zeros((size, size, 3), np.uint8)
        vis = (col >= min_hits)
        if vis.any():
            small = (np.clip(col.astype(np.float32) / max(min_hits * 4, 1),
                             0, 1) * 255).astype(np.uint8)
            small[~vis] = 0
            big = cv2.resize(small, (size, size),
                             interpolation=cv2.INTER_NEAREST)
            img = cv2.applyColorMap(big, cv2.COLORMAP_VIRIDIS)
            img[cv2.resize(vis.astype(np.uint8), (size, size),
                           interpolation=cv2.INTER_NEAREST) == 0] = (20, 20, 20)

        # The origin, and the direction the head faced when it started.
        c = size // 2
        cv2.drawMarker(img, (c, c), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.line(img, (c, c), (c, c - 40), (0, 0, 255), 1)
        cv2.putText(img, f"+Z  ({self.extent / 1000:.1f} m half-width)",
                    (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 180, 180), 1)
        return img


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default="calibration/stereo_calib.npz")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--exposure", type=int, default=None)
    ap.add_argument("--gain", type=int, default=None)
    ap.add_argument("--left-cam", type=int, default=0,
                    help="Picamera2 number of the LEFT eye. Cover the left "
                         "lens and check the left pane darkens; a swapped "
                         "pair gives negative disparity everywhere while "
                         "both views look fine.")
    ap.add_argument("--right-cam", type=int, default=1)

    ap.add_argument("--num-disparities", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--no-wls", action="store_true")
    ap.add_argument("--min-depth-mm", type=float, default=300.0)
    ap.add_argument("--max-depth-mm", type=float, default=2500.0,
                    help="beyond this a 20 mm cell is smaller than the "
                         "depth uncertainty; see the module docstring")

    ap.add_argument("--voxel-mm", type=float, default=20.0)
    ap.add_argument("--extent-mm", type=float, default=3000.0,
                    help="half-width of the grid in X and Z")
    ap.add_argument("--grid-height-mm", type=float, default=1500.0,
                    help="half-height of the grid in Y")
    ap.add_argument("--min-hits", type=int, default=3,
                    help="how many times a cell must be hit before it is "
                         "drawn as occupied")
    ap.add_argument("--stride", type=int, default=2,
                    help="subsample points before integrating; 2 is four "
                         "times fewer points and barely changes the grid")
    ap.add_argument("--every", type=int, default=2,
                    help="integrate every Nth frame. Consecutive frames "
                         "during a slow pan are nearly identical, so "
                         "fusing all of them costs time and adds nothing")
    a = ap.parse_args()

    calib_path = Path(a.calib)
    if not calib_path.exists():
        sys.exit(f"no calibration at {calib_path}")

    settings = CameraSettings(width=a.width, height=a.height, fps=a.fps)
    if a.exposure is not None:
        settings = CameraSettings(width=a.width, height=a.height, fps=a.fps,
                                  exposure_us=a.exposure * 100)
    cam = SyncedStereo(left_num=a.left_cam, right_num=a.right_cam,
                       settings=settings)
    cam.start()
    cap_w, cap_h = a.width, a.height

    rect = Rectifier(calib_path, (cap_w, cap_h))
    geom = rect
    Q = rect.Q
    fx = rect.fx
    baseline = rect.baseline

    print(f"[cam] {cap_w}x{cap_h}  skew tracked per frame")
    print(f"[geom] {rect.describe()}")
    print(f"[frame] origin = baseline midpoint, +X right +Y down +Z forward")
    print(f"        (shifting {baseline / 2:.1f} mm from the left lens)")

    matcher, right_matcher, wls, num_disp = build_sgbm(
        a.num_disparities, a.block_size, not a.no_wls)
    near = rect.fB / max(num_disp - 1.0, 1.0)
    print(f"[sgbm] {num_disp} disparities -> nothing closer than "
          f"{near:.0f} mm")
    for zz in (1000.0, 2000.0, a.max_depth_mm):
        print(f"        at {zz:5.0f} mm: +/-{0.25 * zz * zz / rect.fB:5.1f} mm "
              f"vs {a.voxel_mm:.0f} mm cells")

    grid = VoxelGrid(a.voxel_mm, a.extent_mm, a.grid_height_mm)
    tracker = PoseTracker(Q)

    min_disp = max(1.0, rect.fB / a.max_depth_mm)
    max_disp = min(float(num_disp - 1), rect.fB / a.min_depth_mm)

    print("\nSPACE start/pause | R reset | D view | S save | Q quit")
    print("Rotate SLOWLY and keep the pivot as fixed as you can.\n")

    cv2.namedWindow("grid", cv2.WINDOW_NORMAL)
    view = 0
    running = False
    frame_no = 0
    added_total = 0
    cloud_pts: list = []
    cloud_cols: list = []
    t_last = time.time()
    fps_est = 0.0

    try:
        while True:
            raw_l, raw_r = cam.capture()
            left, right = rect.rectify(raw_l, raw_r)
            gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            frame_no += 1
            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 / max(now - t_last, 1e-6)
            t_last = now

            # SGBM every frame now, because the tracker needs 3D points
            # and not just image shift. That is the price of measuring
            # the pose properly instead of assuming it.
            raw = matcher.compute(gl, gr)
            if wls is not None:
                raw = wls.filter(raw, left, None,
                                 right_matcher.compute(gr, gl))
            disp = raw.astype(np.float32) / 16.0
            valid = (disp >= min_disp) & (disp <= max_disp)

            pose = tracker.update(gl, disp, valid)

            if running and frame_no % a.every == 0:
                vsel = valid.copy()
                if a.stride > 1:
                    thin = np.zeros_like(vsel)
                    thin[::a.stride, ::a.stride] = True
                    vsel &= thin
                xyz = cv2.reprojectImageTo3D(
                    np.where(vsel, disp, 1.0).astype(np.float32), Q)
                z = xyz[:, :, 2]
                vsel &= np.isfinite(z) & (z > a.min_depth_mm) \
                    & (z < a.max_depth_mm)
                pts = xyz[vsel]
                if len(pts):
                    pts[:, 0] -= baseline / 2.0
                    # One 4x4 instead of a yaw about an assumed axis:
                    # rotation and translation, about whatever axis the
                    # motion actually used.
                    pts = pts @ pose[:3, :3].T + pose[:3, 3]
                    added_total += grid.add(pts)
                    cloud_pts.append(pts.astype(np.float32))
                    cloud_cols.append(left[vsel])

            # ------------------------------------------------- display
            if view == 0:
                shown = left.copy()
            elif view == 1:
                finite = disp[np.isfinite(disp) & (disp > 0)]
                hi = float(np.percentile(finite, 99)) if finite.size else 1.0
                shown = cv2.applyColorMap(
                    np.clip(disp / max(hi, 1e-3) * 255, 0, 255
                            ).astype(np.uint8), cv2.COLORMAP_TURBO)
            elif view == 2:
                shown = grid.topdown(a.min_hits)
            else:
                shown = left.copy()

            occ = grid.occupied(a.min_hits)
            lines = [
                f"{'RUNNING' if running else 'paused '}   "
                f"yaw {tracker.yaw_deg:+6.1f}  pitch {tracker.pitch_deg:+5.1f}"
                f"  moved {tracker.translation_mm:5.0f} mm  "
                f"{fps_est:4.1f} fps",
                f"{occ} cells occupied   {added_total} points integrated",
                f"skew {cam.skew_ms * 1000:.0f} us   "
                f"track {'ok' if tracker.last_ok else 'LOST'}  "
                f"{tracker.last_inliers}/{tracker.last_n} inliers  "
                f"{tracker.commits} commits",
            ]
            if tracker.last_ok and tracker.last_inliers < 30:
                lines.append("!! few inliers -- pose is unreliable here")

            for i, t in enumerate(lines):
                for th, c in ((3, (0, 0, 0)), (1, (0, 255, 0))):
                    cv2.putText(shown, t, (8, 20 + i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, th,
                                cv2.LINE_AA)

            cv2.imshow("grid", shown)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord(" "):
                running = not running
                print(f"[grid] {'running' if running else 'paused'} at "
                      f"{tracker.yaw_deg:+.1f} deg")
            elif k == ord("r"):
                grid.grid[:] = 0
                tracker.reset()
                cloud_pts.clear()
                cloud_cols.clear()
                added_total = 0
                print("[grid] reset -- this pose is now the origin")
            elif k == ord("d"):
                view = (view + 1) % 3
            elif k == ord("s"):
                np.savez_compressed(
                    "grid.npz", grid=grid.grid,
                    voxel_mm=np.array(a.voxel_mm),
                    extent_mm=np.array(a.extent_mm),
                    height_mm=np.array(a.grid_height_mm))
                if cloud_pts:
                    P = np.concatenate(cloud_pts)
                    C = np.concatenate(cloud_cols)
                    with open("cloud.ply", "w") as f:
                        f.write("ply\nformat ascii 1.0\n")
                        f.write(f"element vertex {len(P)}\n")
                        f.write("property float x\nproperty float y\n"
                                "property float z\n")
                        f.write("property uchar red\nproperty uchar green\n"
                                "property uchar blue\nend_header\n")
                        for p, c in zip(P, C):
                            f.write(f"{p[0]:.1f} {p[1]:.1f} {p[2]:.1f} "
                                    f"{c[2]} {c[1]} {c[0]}\n")
                    print(f"[save] grid.npz + cloud.ply ({len(P)} points)")
                else:
                    print("[save] grid.npz (no cloud yet)")
    finally:
        cam.close()
        cv2.destroyAllWindows()
        print(f"\n{grid.occupied(a.min_hits)} cells occupied, "
              f"final angle {tracker.yaw_deg:+.1f} deg")


if __name__ == "__main__":
    main()
