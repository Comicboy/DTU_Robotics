import cv2
import numpy as np
import os

# ----------------------------
# PARAMETRI DELLA SCACCHIERA
# ----------------------------
corner_cols = 9   # orizzontale (spigoli interni)
corner_rows = 6   # verticale
square_size = 25  # mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# punti 3D ideali
objp = np.zeros((corner_rows * corner_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_cols, 0:corner_rows].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

# ----------------------------
# IMMAGINI OTTIME SELEZIONATE
# ----------------------------
good_images = [
    "Calibration/img_005.jpg",
    "Calibration/img_006.jpg",
    "Calibration/img_007.jpg",
    "Calibration/img_008.jpg"
]

print("Userò SOLO le seguenti immagini per la calibrazione:")
for g in good_images:
    print(" -", g)

# ----------------------------
# CARICAMENTO + CORNERS
# ----------------------------
for fname in good_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (corner_cols, corner_rows), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        print(f"[OK] Corner trovati in {fname}")
    else:
        print(f"[ERRORE] Corner NON trovati in {fname} — controllare l’immagine")

# ----------------------------
# CALIBRAZIONE
# ----------------------------
print("\nCalibrazione in corso...")
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n=== RISULTATI CALIBRAZIONE ===")
print("Matrice intrinseca K:\n", K)
print("\nCoefficienti di distorsione:\n", dist.ravel())
print("\nErrore medio di reproiezione:", ret)

np.savez("calibration_data.npz", K=K, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("\nDati salvati in 'calibration_data.npz'")

# ----------------------------
# UNDISTORT DI ESEMPIO
# ----------------------------
os.makedirs("Calibration/undistorted", exist_ok=True)

for i, fname in enumerate(good_images):
    img = cv2.imread(fname)
    und = cv2.undistort(img, K, dist)
    out_path = f"Calibration/undistorted/und_{i}.png"
    cv2.imwrite(out_path, und)
    print(f"Immagine corretta salvata: {out_path}")
