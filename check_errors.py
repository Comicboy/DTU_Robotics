import cv2, numpy as np, glob, os

corner_cols = 9
corner_rows = 6
square_size = 25.0  # mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((corner_rows*corner_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_cols, 0:corner_rows].T.reshape(-1, 2)
objp *= square_size

data = np.load("calibration_data.npz")
K = data["K"]
dist = data["dist"]

images = sorted(glob.glob("Calibration/*.jpg") + glob.glob("Calibration/*.png"))
os.makedirs("Calibration/overlays", exist_ok=True)

errors = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (corner_cols, corner_rows), None)
    if not ret:
        continue

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    ok, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)

    err = np.sqrt(np.mean(np.sum((corners2.reshape(-1,2) - proj)**2, axis=1)))

    errors.append((err, fname))

    vis = img.copy()
    for p in corners2.reshape(-1, 2):
        cv2.circle(vis, tuple(p.astype(int)), 4, (0,255,0), -1)
    for p in proj:
        cv2.circle(vis, tuple(p.astype(int)), 3, (0,0,255), 1)
    cv2.putText(vis, f"err={err:.2f}px", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imwrite(f"Calibration/overlays/{os.path.basename(fname)}", vis)

errors.sort(reverse=True)
print("\n--- ERRORI PER IMMAGINE ---")
for e,f in errors:
    print(f"{e:.3f} px : {f}")
