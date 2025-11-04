import cv2
import numpy as np

# --- PARAMETERS ---
# TODO: Figure out how to get the frame from the robot camera?
CAMERA_MODE = False  # True for webcam, False for image
IMAGE_PATH = "keyboard.jpg"

# --- READ IMAGE OR CAMERA ---
if CAMERA_MODE:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Cannot access camera")
else:
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise ValueError("Cannot read image")

# Resize for visibility
frame = cv2.resize(frame, (1000, 400))

# --- PREPROCESSING ---
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# --- FIND CONTOURS ---
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

key_centers = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # filter unwanted contours
    if w < 20 or h < 20 or w > 150 or h > 150:
        continue
    aspect_ratio = w / float(h)
    if 0.7 < aspect_ratio < 1.5:
        cx, cy = x + w // 2, y + h // 2
        key_centers.append((cx, cy, x, y, w, h))

# Sort roughly top-to-bottom, left-to-right
key_centers = sorted(key_centers, key=lambda p: (p[1], p[0]))

# --- MANUAL LABELING ---
# First row of a QWERTY keyboard
manual_labels = list("QWERTYUIOPASDFGHJKLZXCVBNM1234567890")  # modify as needed

# Create mapping: label → (cx, cy)
key_map = {}
for i, (cx, cy, x, y, w, h) in enumerate(key_centers):
    if i < len(manual_labels):
        label = manual_labels[i]
    else:
        label = f"?{i}"
    key_map[label] = (cx, cy)
    # draw on frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x + 5, y + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

print("Detected and mapped keys:")
for k, v in key_map.items():
    print(f"  {k}: {v}")

# Save mapping to file (for reuse later)
np.save("keyboard_keymap.npy", key_map)

cv2.imshow("Keyboard Key Mapping", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
