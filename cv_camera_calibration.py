import numpy as np
import cv2
import glob
import json
import os

# =========================================================
# CONFIGURATION
# =========================================================

# 1. SQUARE SIZE (Meters)
# 25mm = 0.025m
SQUARE_SIZE = 0.025 

# 2. INNER CORNERS
# 9 squares horizontal -> 8 inner corners
# 7 squares vertical   -> 6 inner corners
NB_HORIZONTAL = 8
NB_VERTICAL = 6

# =========================================================
# PREPARATION
# =========================================================

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((NB_HORIZONTAL * NB_VERTICAL, 3), np.float32)
objp[:, :2] = np.mgrid[0:NB_VERTICAL, 0:NB_HORIZONTAL].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# --- ABSOLUTE PATH CONFIGURATION ---
image_folder = r'c:/Users/peisz/OneDrive/Documents/dtu_workspace/DTU_Robotics/calibration_images'
search_path = os.path.join(image_folder, '*.jpg')

images = glob.glob(search_path)

if not images:
    print(f"ERROR: No images found at: {search_path}")
    print(f"Checking folder: {image_folder}")
    print("Please verify the path is correct and contains .jpg files.")
    exit()

print(f"Found {len(images)} images. Starting processing...")

# =========================================================
# CORNER DETECTION LOOP
# =========================================================

valid_images = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load {os.path.basename(fname)}")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # Flags: Adaptive Threshold (lighting), Normalize (contrast), Fast Check (speed)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, (NB_VERTICAL, NB_HORIZONTAL), flags)

    if ret == True:
        objpoints.append(objp)

        # Refine for sub-pixel accuracy
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_refined)

        # Visualization
        vis_img = img.copy()
        cv2.drawChessboardCorners(vis_img, (NB_VERTICAL, NB_HORIZONTAL), corners_refined, ret)
        
        # Resize for display
        display_h = 600
        scale = display_h / vis_img.shape[0]
        vis_img = cv2.resize(vis_img, (0,0), fx=scale, fy=scale) 
        
        cv2.imshow('Calibration Detection', vis_img)
        cv2.waitKey(50) # Fast playback
        valid_images += 1
        print(f"[{valid_images}] Used: {os.path.basename(fname)}")
    else:
        print(f"  [X] Skipped: {os.path.basename(fname)} (Corners not found)")

cv2.destroyAllWindows()

if valid_images < 10:
    print(f"\nWARNING: Only {valid_images} valid images. Calibration might be poor.")
    if valid_images == 0:
        exit()

# =========================================================
# CALIBRATION
# =========================================================

print(f"\nCalibrating camera with {valid_images} images...")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n" + "="*40)
print(f"CALIBRATION RESULT")
print("="*40)
print(f"RMS Re-projection Error: {ret:.4f} pixels")
print("-" * 40)
print("Camera Matrix (K):")
print(mtx)
print("-" * 40)
print("Distortion Coefficients (D):")
print(dist.ravel())
print("="*40)

# =========================================================
# UNDISTORTION CHECK
# =========================================================

print("Showing undistortion check on the first image...")
img = cv2.imread(images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Resize for display
display_h = 600
scale = display_h / img.shape[0]
img_small = cv2.resize(img, (0,0), fx=scale, fy=scale)
dst_small = cv2.resize(dst, (0,0), fx=scale, fy=scale)

cv2.imshow('Original (Scaled)', img_small)
cv2.imshow('Undistorted (Scaled)', dst_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =========================================================
# SAVE
# =========================================================

# Save to the script directory
output_file = os.path.join(r'c:/Users/peisz/OneDrive/Documents/dtu_workspace/DTU_Robotics', "camera_calibration.json")

data = {
    "K": mtx.tolist(),
    "D": dist.tolist(),
    "rms_error": ret,
    "image_width": img.shape[1],
    "image_height": img.shape[0]
}

with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"\nSaved calibration to '{output_file}'")