import numpy as np
import cv2
import glob
import json
import os

# =========================================================
# CONFIGURATION
# =========================================================

# 1. SQUARE SIZE
# Value is in METERS (25mm = 0.025m)
# This ensures your translation vectors (T) are in meters.
SQUARE_SIZE = 0.025 

# 2. INNER CORNERS (Not number of squares!)
# You have 9 squares horizontal -> 8 inner corners
# You presumably have 7 squares vertical -> 6 inner corners
# If detection fails, try swapping these to (6, 8)
NB_HORIZONTAL = 8
NB_VERTICAL = 6

# =========================================================
# PREPARATION
# =========================================================

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...
# We multiply by SQUARE_SIZE to convert "grid units" to "meters"
objp = np.zeros((NB_HORIZONTAL * NB_VERTICAL, 3), np.float32)
objp[:, :2] = np.mgrid[0:NB_VERTICAL, 0:NB_HORIZONTAL].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Load images
# Make sure your images are in a folder named 'imgs' relative to this script
images = glob.glob('imgs/*.png')

if not images:
    print("ERROR: No images found in 'imgs/' folder.")
    print("Please check the path or extension.")
    exit()

print(f"Found {len(images)} images. Starting processing...")

# =========================================================
# CORNER DETECTION LOOP
# =========================================================

valid_images = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # Using adaptive thresholding helps with uneven lighting
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, (NB_VERTICAL, NB_HORIZONTAL), flags)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refine the corner detection for sub-pixel accuracy
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_refined)

        # Draw and display the corners
        # (We draw on a copy to keep the loop fast and clean)
        vis_img = img.copy()
        cv2.drawChessboardCorners(vis_img, (NB_VERTICAL, NB_HORIZONTAL), corners_refined, ret)
        
        cv2.imshow('Calibration Detection', vis_img)
        cv2.waitKey(100) # Wait 100ms per image to visualize
        valid_images += 1
    else:
        print(f"Corners not found in {fname} - Check lighting or grid size settings.")

cv2.destroyAllWindows()

if valid_images == 0:
    print("Error: No valid corners detected in any image.")
    exit()

# =========================================================
# CALIBRATION
# =========================================================

print(f"\nCalibrating on {valid_images} valid images...")

# mtx: Camera Matrix (Intrinsic)
# dist: Distortion Coefficients
# rvecs, tvecs: Rotation and Translation vectors for each image
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n" + "="*30)
print(f"CALIBRATION SUCCESSFUL")
print("="*30)
print(f"RMS Re-projection Error: {ret:.4f} pixels")
print("-" * 30)
print(f"Camera Matrix (K):\n{mtx}")
print("-" * 30)
print(f"Distortion Coeffs (D):\n{dist}")
print("="*30)

# =========================================================
# VISUAL VERIFICATION (Undistortion)
# =========================================================

print("Displaying undistortion result on the first valid image...")
img = cv2.imread(images[0])
h, w = img.shape[:2]

# Get optimal camera matrix (refines the view to remove black borders if needed)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Crop the image based on the valid region of interest (ROI)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

cv2.imshow('Original', img)
cv2.imshow('Undistorted', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =========================================================
# SAVE TO JSON
# =========================================================

output_file = "camera_calibration.json"
data = {
    "K": mtx.tolist(),
    "D": dist.tolist(),
    "rms_error": ret,
    "image_width": img.shape[1],
    "image_height": img.shape[0]
}

with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"\nCalibration data saved to '{output_file}'")