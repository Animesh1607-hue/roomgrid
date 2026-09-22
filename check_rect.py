import numpy as np, cv2
from stereo_camera import SyncedStereo, CameraSettings

size = (640, 480)
d = np.load("stereo_calib.npz")
R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(d["K1"],d["D1"],d["K2"],d["D2"],
    size,d["R"],d["T"],flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
mL = cv2.initUndistortRectifyMap(d["K1"],d["D1"],R1,P1,size,cv2.CV_16SC2)
mR = cv2.initUndistortRectifyMap(d["K2"],d["D2"],R2,P2,size,cv2.CV_16SC2)

with SyncedStereo(settings=CameraSettings()) as cam:
    for _ in range(5): L,R = cam.capture()

cv2.imwrite("raw_left.png", L); cv2.imwrite("raw_right.png", R)
rL = cv2.remap(L,*mL,cv2.INTER_LINEAR); rR = cv2.remap(R,*mR,cv2.INTER_LINEAR)
pair = np.hstack([rL,rR])
for y in range(0,480,40):
    cv2.line(pair,(0,y),(pair.shape[1],y),(0,255,0),1)
cv2.imwrite("rect_pair.png", pair)
print("wrote raw_left.png raw_right.png rect_pair.png")
print("mean brightness L/R:", L.mean(), R.mean())
