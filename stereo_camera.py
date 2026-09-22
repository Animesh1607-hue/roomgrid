"""
Two Arducam B0445 (IMX296) on a Pi 5, frame-synchronised.

Standalone: this package does not import from Stereo-Vision-Humanoid, so
it can be copied anywhere on the Pi and run. The camera handling below is
the same approach that measured 19 us median skew on this rig.


WHY THE SYNC MATTERS AT ALL
---------------------------
Two free-running IMX296 have independent oscillators. Measured before
sync was enabled, the gap between the two frames wandered from +18.8 ms
to -12.5 ms -- the SIGN flipped, so it is not a fixed offset that could
be calibrated away. It is uniform-random across the frame period and
unknowable per frame.

That gap becomes a horizontal displacement on anything moving, and
horizontal displacement is exactly what a stereo matcher reads as
disparity. No error, no dropped frame, no symptom -- just wrong depth on
precisely the moving objects that matter.

libcamera's software sync fixes it. One camera is the server and
broadcasts timing; the other is a client and stretches or shrinks its
frame durations to converge. Same sensor on the same Pi converges to a
few tens of microseconds.

THREE CONSEQUENCES, all load-bearing:

1. FRAME DURATION BELONGS TO THE SYNC ALGORITHM. It converges by
   adjusting frame duration, so writing FrameDurationLimits from
   application code fights it and sync never settles. Frame rate is
   fixed once, in the configuration, and never touched again -- which
   caps exposure just under the frame period.

2. THE CLIENT STARTS FIRST. It waits, idle, until the server appears.
   Starting the server first works but wastes the first few seconds on
   unsynchronised frames.

3. 3A STAYS OFF. There is no cross-camera AE/AWB arbitration in
   libcamera, so two auto loops on one scene reach different answers and
   the matcher loses correspondences that no longer look alike. Every
   control is written to both cameras from one variable.


SENSOR MODE IS PINNED
---------------------
sensor={"output_size": (1456, 1088)} forces the full sensor read with ISP
downscale rather than letting libcamera choose. Left free it may pick a
cropped mode on some runs, which changes both the field of view and fx --
silently invalidating a calibration captured under the other mode.


RGB888 IS BGR
-------------
Picamera2's "RGB888" returns BGR byte order in the numpy array, which is
what OpenCV wants. Looks wrong, is right.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

SENSOR_NATIVE = (1456, 1088)


@dataclass(frozen=True)
class CameraSettings:
    width: int = 640
    height: int = 480
    fps: float = 30.0

    # Short exposure with more gain, deliberately. The matcher tolerates
    # noise far better than blur, and a panning head blurs everything.
    exposure_us: int = 14000
    analogue_gain: float = 3.0
    colour_gains: tuple[float, float] = (2.78, 1.90)

    @property
    def frame_us(self) -> int:
        return int(round(1_000_000 / self.fps))

    @property
    def max_exposure_us(self) -> int:
        return self.frame_us - 3000

    def controls(self) -> dict:
        exp = min(int(self.exposure_us), self.max_exposure_us)
        return {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": exp,
            "AnalogueGain": float(self.analogue_gain),
            "ColourGains": tuple(float(g) for g in self.colour_gains),
        }


class SyncedStereo:
    """Two CSI cameras presented as one synchronised source."""

    def __init__(self, left_num=1, right_num=0, settings=None):
        from picamera2 import Picamera2

        self.settings = settings or CameraSettings()
        info = Picamera2.global_camera_info()
        if len(info) < 2:
            raise RuntimeError(
                f"{len(info)} camera(s) detected, need 2. Check both "
                f"ribbon cables and that config.txt has BOTH "
                f"dtoverlay=imx296,cam0 and dtoverlay=imx296,cam1 -- two "
                f"lines without distinct cam numbers puts both sensors on "
                f"the same port and one is silently ignored."
            )

        self._left = self._open(left_num, server=True)
        self._right = self._open(right_num, server=False)
        self._skew: list[float] = []
        self.last_sensor_ts_ns = 0

    def _open(self, num, server):
        from libcamera import controls
        from picamera2 import Picamera2

        try:
            mode = (controls.rpi.SyncModeEnum.Server if server
                    else controls.rpi.SyncModeEnum.Client)
        except AttributeError as e:
            raise RuntimeError(
                "this libcamera has no SyncMode control -- update the "
                "camera stack (sudo apt full-upgrade, then reboot). "
                "Without it the pair free-runs and depth on anything "
                "moving is unreliable."
            ) from e

        cam = Picamera2(num)
        cam.configure(cam.create_video_configuration(
            main={"size": (self.settings.width, self.settings.height),
                  "format": "RGB888"},
            sensor={"output_size": SENSOR_NATIVE},
            controls={"FrameRate": self.settings.fps, "SyncMode": mode},
            buffer_count=6))
        return cam

    def start(self, settle_s=1.0):
        self._right.start()                 # client first -- see docstring
        self._left.start()
        self._left.set_controls(self.settings.controls())
        self._right.set_controls(self.settings.controls())

        print("[cam] waiting for sync convergence...")
        for cam in (self._right, self._left):
            cam.capture_sync_request().release()
        time.sleep(settle_s)
        print("[cam] synchronised")

    def capture(self):
        """(left, right) as BGR arrays. Skew is tracked internally."""
        req_l = self._left.capture_request()
        req_r = self._right.capture_request()
        try:
            left = req_l.make_array("main")
            right = req_r.make_array("main")
            ts_l = req_l.get_metadata().get("SensorTimestamp", 0)
            ts_r = req_r.get_metadata().get("SensorTimestamp", 0)
        finally:
            req_l.release()
            req_r.release()

        self.last_sensor_ts_ns = int(ts_l)
        self._skew.append(abs(ts_l - ts_r) / 1e6)
        if len(self._skew) > 200:
            self._skew.pop(0)
        return left, right

    @property
    def skew_ms(self) -> float:
        """Median |skew| in ms. Log it: a mount that takes a knock shows
        up here as a number before it shows up as bad depth."""
        return float(np.median(self._skew)) if self._skew else 0.0

    def close(self):
        for cam in (self._left, self._right):
            try:
                cam.stop()
                cam.close()
            except Exception:                              # noqa: BLE001
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
        return False
