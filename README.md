# roomgrid

Build a 3D occupancy grid of the room by sweeping a stereo head across it.

Standalone — nothing here imports from Stereo-Vision-Humanoid. Copy the
folder anywhere on the Pi, point it at a calibration file, and run.

```
scp -r roomgrid usvz@rs04pi.local:~/Desktop/
ssh usvz@rs04pi.local
cd ~/Desktop/roomgrid
cp ~/Desktop/Stereo-Vision-Humanoid/calibration/stereo_calib.npz .
python3 build_grid.py --calib stereo_calib.npz
```

Needs `python3-picamera2`, `python3-opencv` (the apt build, for
`ximgproc`) and numpy. All already on the Pi.


## What it does

The instant you press R, wherever the head is pointing becomes the world.
Origin sits midway between the two lenses; +X right, +Y down, +Z forward,
matching OpenCV so nothing needs converting.

**That frame then stops moving.** Pan the head and the cameras travel
through it. Each frame's point cloud is measured from wherever the head
is now, so it is rotated back by the running angle estimate before being
dropped into the grid. A chair 1.2 m ahead keeps the same coordinates
whether it was seen at −40° or +10°.

Space is cut into cubes of `--voxel-mm`. Any cube a point lands in is
occupied. Sweep, and the cubes fill in across the whole arc.

The result is a block of space around the robot where every cube is
known-occupied, known-empty, or unobserved — so the arm can be told
"reach to cell (60, 12, 40)" rather than "reach toward that thing".


## Keys

```
SPACE   start / pause accumulating
R       reset the grid and zero the angle -- this pose becomes the origin
D       cycle view: live | disparity | top-down grid
S       save grid.npz and cloud.ply
Q       quit
```


## The weak part, stated plainly

With no encoder the rotation is estimated from the images. A pure yaw of
dθ shifts a feature horizontally by `fx · dθ` regardless of its depth,
and that depth independence is what makes it work with no knowledge of
the scene. ORB features, matched frame to frame, median shift,
accumulated.

Two errors follow from turning it by hand:

**The pivot wanders.** A motor turns about a fixed axis; a hand does not,
so every frame carries an unknown translation as well as a rotation. The
code assumes pure rotation and cannot tell the difference, so hand
translation is silently absorbed as extra angle. The grid smears by
however far your hand moved.

**Drift accumulates.** Each estimate carries perhaps 0.2°, and they add.
Over 180° in 40 frames that is a few degrees by the far end; at 2 m a 2°
error is 70 mm, more than three voxels. The far end of a sweep will be
visibly worse than the near end.

Resting the rig on anything that pivots about a fixed vertical axis — a
tripod head, a lazy susan, a bolt through a plate — removes most of the
first. An encoder removes both, and is the right answer once the head
actuator is in place: read the pan angle, replace the estimate, done.

**Watch the `tilt` number.** It is the median vertical feature motion,
near zero for a pure pan. Above ~4 px the rig is being carried rather
than turned, and it says so.


## The test that matters

Sweep past a flat wall with the top-down view up (**D** twice). A
straight wall should come out as a straight line. If it curves, that
curve is your drift, and you can see how much.

Then the repeatability check: put a known object where both ends of the
sweep can see it. If it occupies the same cells at −30° and +30°, the
geometry is holding.


## Range, and where the grid stops being honest

Depth error grows with the **square** of distance, because disparity
shrinks as 1/Z while subpixel uncertainty stays put. At fB ≈ 54000 a
quarter pixel is:

| distance | error | vs a 20 mm cell |
|---|---|---|
| 1 m | ±4.6 mm | inside one cell |
| 2 m | ±19 mm | about one cell |
| 3 m | ±42 mm | two cells |
| 5 m | ±116 mm | six cells |

So 20 mm cells mean something out to roughly 2 m and are increasingly
decorative beyond. `--max-depth-mm` defaults to 2500 for that reason.
Raise it if you want the walls in the grid, but do not trust cell
boundaries out there.


## Files

```
build_grid.py     the sweep, the tracker, the grid, the viewer
stereo_camera.py  synchronised capture from the two IMX296
```

`stereo_camera.py` is the piece that gets the two sensors to within a few
tens of microseconds of each other. Before sync was enabled on this rig
the gap wandered from +18.8 ms to −12.5 ms — the sign flipped, so it was
not a fixed offset that could be calibrated away. That gap becomes
horizontal displacement on anything moving, and horizontal displacement
is exactly what a matcher reads as disparity.


## Next step

Replace the visual angle estimate with the actuator's encoder. One change
in `main()`: instead of `tracker.update(gl)`, read the pan angle over CAN.
Everything downstream is unchanged, and both error sources above go away.

The camera then sits some distance forward of the pan axis, so the
transform gains a translation as well as a rotation — measure the offset
from the axis to the baseline midpoint and it drops straight in.
