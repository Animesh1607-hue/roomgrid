#!/usr/bin/env python3
"""
Wall Capture -- burst-average stereo depth of a flat wall.

Builds on stereo_camera.SyncedStereo (hardware-synced IMX296 pair).
Output goes to captures/wall_<timestamp>/ for the plane fit to consume.

UNITS: T in stereo_calib.npz is in the same units as square_size (44.0),
so everything here is in MILLIMETRES. meta.json records this so nothing
downstream has to guess.

CAMERA MAPPING: left = cam1, right = cam0.
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import cv2

from stereo_camera import SyncedStereo, CameraSettings


def load_calib(path, size):
    d = np.load(path)
    K1, D1 = d["K1"], d["D1"]
    K2, D2 = d["K2"], d["D2"]
    R, T = d["R"], d["T"]
    sq = float(d["square_size"]) if "square_size" in d.files else None

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    mapL = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
    mapR = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2)
    return dict(mapL=mapL, mapR=mapR, Q=Q, P1=P1,
                baseline=abs(float(T.ravel()[0])), square_size=sq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="stereo_calib.npz")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--exposure", type=int, default=14000)
    ap.add_argument("--gain", type=float, default=3.0)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--num-disp", type=int, default=96)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--min-valid-frac", type=float, default=0.6)
    ap.add_argument("--max-std", type=float, default=1.0)
    ap.add_argument("--out", default="captures")
    args = ap.parse_args()

    size = (args.width, args.height)
    cal = load_calib(args.calib, size)
    fx = float(cal["P1"][0, 0])
    b = cal["baseline"]
    unit = "mm" if b > 1.0 else "m"
    print(f"baseline {b:.3f} {unit} | fx {fx:.1f} px | square_size {cal['square_size']}")
    print(f"nearest measurable at disp {args.num_disp}: {fx*b/args.num_disp:.1f} {unit}")

    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=args.num_disp, blockSize=args.block,
        P1=8*args.block**2, P2=32*args.block**2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    settings = CameraSettings(width=args.width, height=args.height,
                              fps=args.fps, exposure_us=args.exposure,
                              analogue_gain=args.gain)

    stack, last_rect = [], None
    win = "wall capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 520)
    fxb = fx * b                                   # depth = fx * baseline / disparity
    with SyncedStereo(settings=settings) as cam:
        print("live view: aim at the target, SPACE = start capture, Q = quit")
        capturing = False
        t0 = time.time()
        while True:
            L, R = cam.capture()
            rL = cv2.remap(L, *cal["mapL"], cv2.INTER_LINEAR)
            rR = cv2.remap(R, *cal["mapR"], cv2.INTER_LINEAR)
            gL = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY)
            gR = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)
            disp = matcher.compute(gL, gR).astype(np.float32) / 16.0
            valid = disp > 0
            h, w = disp.shape
            c = disp[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20]
            c = c[c > 0]
            centre = fxb / np.median(c) if c.size > 20 else float("nan")

            d8 = np.clip(disp / args.num_disp * 255, 0, 255).astype(np.uint8)
            vis = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
            vis[~valid] = 0
            view = np.hstack([rL.copy(), vis])
            for x0 in (0, w):
                cv2.rectangle(view, (x0 + w // 2 - 20, h // 2 - 20),
                              (x0 + w // 2 + 20, h // 2 + 20), (255, 255, 255), 1)
            state = (f"CAPTURING {len(stack)}/{args.frames}" if capturing
                     else "LIVE - SPACE to capture")
            cv2.putText(view, f"{state}   valid {valid.mean() * 100:4.1f}%   "
                        f"centre {centre:6.0f} {unit}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and not capturing:
                capturing, t0 = True, time.time()
                print(f"capturing {args.frames} frames -- hold still")
            if capturing:
                stack.append(disp)
                last_rect = rL
                n = len(stack)
                if n == 1 or n % 5 == 0:
                    print(f"  {n}/{args.frames}  valid {valid.mean() * 100:5.1f}%"
                          f"  {(time.time() - t0) / n:.2f}s/frame")
                if n >= args.frames:
                    break
        skew = cam.skew_ms
    cv2.destroyAllWindows()
    if not stack:
        print("nothing captured")
        return

    D = np.stack(stack)
    valid = D > 0
    frac = valid.mean(axis=0)
    Dm = np.where(valid, D, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nan_to_num(np.nanmean(Dm, axis=0), nan=0.0)
        std  = np.nan_to_num(np.nanstd(Dm,  axis=0), nan=1e9)

    mask = (frac >= args.min_valid_frac) & (std <= args.max_std) & (mean > 0)
    print(f"\nskew {skew:.3f} ms | valid after averaging {mask.mean()*100:.1f}%")
    if mask.mean() < 0.30:
        print("WARNING: low coverage -- the wall needs more texture")

    pts = cv2.reprojectImageTo3D(
        np.where(mask, mean, 0).astype(np.float32), cal["Q"])

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / f"wall_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/"disparity_mean.npy", mean.astype(np.float32))
    np.save(out/"valid_mask.npy", mask)
    np.save(out/"points.npy", pts.astype(np.float32))
    cv2.imwrite(str(out/"left_rect.png"), last_rect)

    z = pts[..., 2][mask]
    if z.size == 0:
        print("\nNo valid pixels at all -- nothing to save.")
        print("Check: wall texture, distance (must exceed nearest-measurable),")
        print("       exposure, and that left/right are not swapped.")
        cv2.imwrite("debug_left_rect.png", last_rect)
        print("wrote debug_left_rect.png -- look at it")
        return
    meta = dict(
        timestamp=ts, frames=len(stack), size=list(size),
        exposure_us=args.exposure, gain=args.gain,
        num_disp=args.num_disp, block=args.block,
        skew_ms=skew, baseline=b, units=unit,
        square_size=cal["square_size"],
        valid_fraction=float(mask.mean()),
        depth_median=float(np.median(z)),
        depth_p5=float(np.percentile(z, 5)),
        depth_p95=float(np.percentile(z, 95)),
        origin="left_rect_camera", Q=cal["Q"].tolist())
    (out/"meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwritten to {out}")
    print(f"median wall depth: {meta['depth_median']:.1f} {unit}")
    print(f"spread p5-p95: {meta['depth_p5']:.1f} .. {meta['depth_p95']:.1f} {unit}")
    print("\nCHECK: median should match your tape measure to the wall.")


if __name__ == "__main__":
    main()
