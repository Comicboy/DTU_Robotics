# =========================================================
# PART 2A: STEREO CAMERA CONFIGURATION
# =========================================================
import numpy as np
K_CALIB = np.eye(3) 
D_CALIB = np.zeros(5)

# 1. YOUR ROBOT MATRIX (Pose 2 in Frame 1, in Robot Coords)
H_ROBOT_RAW = np.array([
    [-3.09016994e-01,  9.51056516e-01,  1.49415645e-33,  1.81895527e+01],
    [-9.51056516e-01, -3.09016994e-01, -5.67574998e-34, -7.04171936e+01],
    [-2.12494080e-33, -4.03425723e-33,  1.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
])

# 2. CONVERT TRANSLATION UNITS (mm -> meters)
H_ROBOT_METERS = H_ROBOT_RAW.copy()
H_ROBOT_METERS[:3, 3] = H_ROBOT_METERS[:3, 3] / 1000.0

# 3. DEFINE THE AXIS MAPPING (Robot -> OpenCV)
# Col 0: Where does Robot X go? -> OpenCV Z (0,0,1)
# Col 1: Where does Robot Y go? -> OpenCV -Y (0,-1,0)
# Col 2: Where does Robot Z go? -> OpenCV -X (-1,0,0)
R_CORR = np.array([
    [ 0,  0, -1],
    [ 0, -1,  0],
    [ 1,  0,  0]
])

# Create 4x4 Correction Matrix
H_CORR = np.eye(4)
H_CORR[:3, :3] = R_CORR

# 4. APPLY CHANGE OF BASIS
# Formula: H_new = H_corr * H_old * inv(H_corr)
H_OPENCV_MOTION = H_CORR @ H_ROBOT_METERS @ np.linalg.inv(H_CORR)

print("Converted Matrix (OpenCV Frame):")
print(H_OPENCV_MOTION)

# 5. INVERT MOTION FOR RECTIFICATION
# OpenCV Rectify needs the coordinate transform (Frame 1 -> Frame 2)
# This is the inverse of the Motion Matrix.
H_FINAL = np.linalg.inv(H_OPENCV_MOTION)

# 6. EXTRACT
R_EXT = H_FINAL[:3, :3]
T_EXT = H_FINAL[:3, 3]

CAM_RES = (1280, 720)