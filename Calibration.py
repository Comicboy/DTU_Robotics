import cv2
import numpy as np
import glob


# ----------------------------
# CHESSBOARD PARAMETERS
# ----------------------------
corner_cols = 6   # horizontal corners
corner_rows = 9   # vertical corners
square_size = 25  # in mm

# refinement criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D points of the chessboard (plane z = 0)
objp = np.zeros((corner_rows*corner_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_cols, 0:corner_rows].T.reshape(-1, 2)
objp *= square_size  # real-world scale

objpoints = []  # 3D points
imgpoints = []  # 2D points

# ----------------------------
# IMAGE LOADING
# ----------------------------
images = glob.glob("BATSA/*.jpg") + glob.glob("BATSA/*.png")

print(f"Found {len(images)} images.")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (corner_cols, corner_rows), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        print(f"[OK] Corners found in {fname}")
    else:
        print(f"[SKIP] No corners found in {fname}")

# ----------------------------
# CALIBRATION
# ----------------------------
print("\nCalibration in progress...")
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n=== CALIBRATION RESULTS ===")
print("Intrinsic matrix K:\n", K)
print("\nDistortion coefficients:\n", dist.ravel())
print("\nAverage reprojection error:", ret)