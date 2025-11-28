import cv2
import numpy as np
import glob
import os

# ----------------------------
# PARAMETRI DELLA SCACCHIERA
# ----------------------------
corner_cols = 6   # orizzontale
corner_rows = 9   # verticale
square_size = 25 # in mm

# criteri per raffinamento
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# punti 3D della scacchiera (piano z=0)
objp = np.zeros((corner_rows*corner_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_cols, 0:corner_rows].T.reshape(-1, 2)
objp *= square_size  # scala reale

objpoints = []  # punti 3D
imgpoints = []  # punti 2D

# ----------------------------
# CARICAMENTO IMMAGINI
# ----------------------------
images = glob.glob("BATSA/*.jpg") + glob.glob("BATSA/*.png")

print(f"Trovate {len(images)} immagini.")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (corner_cols, corner_rows), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        print(f"[OK] Corner trovati in {fname}")
    else:
        print(f"[SKIP] Nessun corner trovato in {fname}")

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

# ----------------------------
# SALVATAGGIO
# ----------------------------
np.savez("calibration_data.npz", K=K, dist=dist, rvecs=rvecs, tvecs=tvecs)

print("\nDati salvati in 'calibration_data.npz'")

# ----------------------------
# ESEMPIO DI UNDISTORT
# ----------------------------
os.makedirs("Calibration/undistorted", exist_ok=True)

for i, fname in enumerate(images[:3]):  # salva solo i primi 3 esempi
    img = cv2.imread(fname)
    und = cv2.undistort(img, K, dist)
    out_path = f"Calibration/undistorted/und_{i}.png"
    cv2.imwrite(out_path, und)
    print(f"Immagine corretta salvata: {out_path}")
