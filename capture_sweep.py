#!/usr/bin/env python3
"""
Capture a pan sweep for DUSt3R, with stereo checkpoints for metric scale.

    python3 capture_sweep.py --out sweeps/sweep_001

    SPACE  save the current LEFT-camera frame as one DUSt3R view
    C      checkpoint -- hold still; gathers a window of frames, fits the
           wall plane the same tested way live_grid.py does, and saves the
           full stereo pair plus the result. This is a metric anchor.
    U      undo the last view or checkpoint
    Q      quit and write manifest.json

Requires live_grid.py in this folder -- its matcher, disparity-offset
handling, and window/plane fitting are used directly, so this shares
exactly the tested pipeline rather than a second copy of it.


WHY LEFT-CAMERA-ONLY VIEWS
---------------------------
DUSt3R takes an ordinary set of images and recovers geometry and camera
pose by comparing structure ACROSS views, not left-right at one instant.
No stereo pairing is needed for that, so only the left camera (cam1) is
saved for the view set. That also means the row corruption a single
extender adds to one frame is not a left-right bias here, the way it is
for SGBM -- multi-view reconstruction is far less sensitive to it.

Panning is keypress-triggered, not motor-triggered, so this works whether
the head is moved by hand or by a commanded actuator -- the script does
not need to know which.


WHY CHECKPOINTS
---------------
DUSt3R's reconstruction is correct in shape but has NO absolute scale.
A checkpoint is a normal stereo capture at that same head position: a
window of frames combined the way capture_wall.py combines a burst, then
fit with live_grid.fit_frame -- the identical, tape-validated pipeline.
Wherever a checkpoint finds a real wall, its axis distance is a metric
number that can be matched against DUSt3R's scale-free distance to the
same surface, giving the one factor that scales the whole reconstruction.
A checkpoint that reports "no wall" is saved anyway, with the reason, so
you can see the sweep found nothing usable there rather than silently
losing it -- try another spot on the same view for the anchor instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_grid import (                                          # noqa: E402
    consistent_disparity, fit_frame, load_rectification, make_matcher,
    tilt_deg,
)


def put_lines(img, lines):
    for i, t in enumerate(lines):
        for thk, c in ((3, (0, 0, 0)), (1, (0, 255, 0))):
            cv2.putText(img, t, (8, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, c, thk, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder, e.g. sweeps/sweep_001")
    ap.add_argument("--calib", default="stereo_calib.npz")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--num-disp", type=int, default=96)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--disp-offset", type=float, default=None,
                    help="defaults to disp_offset saved in the calibration")
    ap.add_argument("--row-filter", type=int, default=3,
                    help="same HDMI row-line filter as live_grid.py; 0 to disable")
    ap.add_argument("--window", type=int, default=12,
                    help="frames combined at a checkpoint, as capture_wall.py "
                         "combines a burst")
    ap.add_argument("--min-valid-frac", type=float, default=0.6)
    ap.add_argument("--max-std", type=float, default=1.0)
    ap.add_argument("--max-tilt", type=float, default=55.0)
    ap.add_argument("--min-inliers", type=int, default=300)
    a = ap.parse_args()

    from stereo_camera import CameraSettings, SyncedStereo

    out = Path(a.out)
    (out / "views").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    rect = load_rectification(a.calib, (a.width, a.height))
    off = rect["offset"] if a.disp_offset is None else a.disp_offset
    fB = rect["fx"] * rect["baseline"]
    matcher = make_matcher(a.num_disp, a.block)
    print(f"fx {rect['fx']:.1f}  baseline {rect['baseline']:.2f} mm  "
          f"offset {off:.2f} px  row filter {a.row_filter or 'off'}")

    def row_filtered(gray):
        if not a.row_filter:
            return gray
        from live_grid import remove_row_lines
        return remove_row_lines(gray, a.row_filter)

    views, checkpoints = [], []
    win = "capture sweep"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"writing to {out}/\n"
          "SPACE save a view | C checkpoint (hold still) | U undo | Q quit")

    with SyncedStereo(settings=CameraSettings(width=a.width,
                                              height=a.height)) as cam:
        checkpointing = False
        stack = deque(maxlen=a.window)
        while True:
            L, R = cam.capture()
            rL = cv2.remap(L, *rect["mL"], cv2.INTER_LINEAR)
            rR = cv2.remap(R, *rect["mR"], cv2.INTER_LINEAR)

            shown = np.hstack([rL, rR]).copy()
            txt = [f"views {len(views)}   checkpoints {len(checkpoints)}"]

            if checkpointing:
                gL = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY)
                gR = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY)
                gLf, gRf = row_filtered(gL), row_filtered(gR)
                disp = matcher.compute(gLf, gRf).astype(np.float32) / 16.0
                if off:
                    disp = np.where(disp > 0, disp - off, disp)
                stack.append(disp)
                txt.append(f"checkpoint: gathering {len(stack)}/{a.window} -- hold still")
                if len(stack) == a.window:
                    mean, trusted = consistent_disparity(
                        list(stack), a.min_valid_frac, a.max_std)
                    fit, note = fit_frame(mean, trusted, rect["Q"], fB,
                                          a.num_disp, a.max_tilt,
                                          a.min_inliers, np.random.default_rng(0))
                    idx = len(checkpoints)
                    cv2.imwrite(str(out / "checkpoints" / f"chk_{idx:03d}_L.png"), rL)
                    cv2.imwrite(str(out / "checkpoints" / f"chk_{idx:03d}_R.png"), rR)
                    if fit is not None:
                        nr, d = fit
                        plane = dict(normal=nr.tolist(), d=float(d),
                                    axis_mm=float(-d / nr[2]),
                                    tilt_deg=tilt_deg(nr), method=note)
                        print(f"checkpoint {idx}: axis {plane['axis_mm']:.0f} mm  "
                              f"tilt {plane['tilt_deg']:.1f} deg  ({note or 'centre'})")
                    else:
                        plane = None
                        print(f"checkpoint {idx}: NO WALL -- {note}")
                    checkpoints.append(dict(
                        index=idx, file_L=f"checkpoints/chk_{idx:03d}_L.png",
                        file_R=f"checkpoints/chk_{idx:03d}_R.png",
                        after_view=len(views), plane=plane, note=note,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))
                    checkpointing = False
                    stack.clear()

            put_lines(shown, txt)
            cv2.imshow(win, shown)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord(" ") and not checkpointing:
                idx = len(views)
                fname = f"view_{idx:03d}.png"
                cv2.imwrite(str(out / "views" / fname), rL)
                views.append(dict(index=idx, file=f"views/{fname}",
                                  timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))
                print(f"view {idx}: saved")
            elif key == ord("c") and not checkpointing:
                checkpointing = True
                stack.clear()
            elif key == ord("u"):
                if checkpointing:
                    checkpointing = False
                    stack.clear()
                    print("checkpoint cancelled")
                elif checkpoints and (not views or
                                     checkpoints[-1]["after_view"] >= len(views)):
                    c = checkpoints.pop()
                    (out / c["file_L"]).unlink(missing_ok=True)
                    (out / c["file_R"]).unlink(missing_ok=True)
                    print(f"undid checkpoint {c['index']}")
                elif views:
                    v = views.pop()
                    (out / v["file"]).unlink(missing_ok=True)
                    print(f"undid view {v['index']}")
    cv2.destroyAllWindows()

    manifest = dict(
        calib=a.calib, width=a.width, height=a.height,
        num_disp=a.num_disp, block=a.block, disp_offset=off,
        row_filter=a.row_filter, views=views, checkpoints=checkpoints,
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n_ok = sum(1 for c in checkpoints if c["plane"] is not None)
    print(f"\n{len(views)} views, {len(checkpoints)} checkpoints "
          f"({n_ok} with a usable wall)")
    if not n_ok:
        print("WARNING: no checkpoint found a wall -- nothing here can anchor "
              "DUSt3R's scale. Add checkpoints where fit_wall.py --verbose "
              "would find a plane before running DUSt3R.")
    print(f"wrote {out}/manifest.json")


if __name__ == "__main__":
    main()