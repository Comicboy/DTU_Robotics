import cv2
import numpy as np

# --- PARAMETERS ---
IMAGE_PATH = "keyboard_image.jpg"

# QWERTY keyboard layout rows
rows = [
    "1234567890",
    "QWERTYUIOP",
    "ASDFGHJKL",
    "ZXCVBNM"
]

# --- READ IMAGE ---
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise ValueError("Cannot read image")
frame = cv2.resize(frame, (1000, 400))

# --- GRAYSCALE + ADAPTIVE THRESHOLD ---
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 3
)

# Morphology to close gaps and remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

# --- FIND CONTOURS ---
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

key_boxes = []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx)

    # Relaxed filter for key-like rectangles
    if 15 < w < 120 and 15 < h < 120:
        aspect_ratio = w/h
        if 0.5 < aspect_ratio < 2.0:
            cx, cy = x + w//2, y + h//2
            key_boxes.append((cx, cy, x, y, w, h))

# --- SORT KEYS ROW-WISE ---
# Step 1: Sort top-to-bottom
key_boxes.sort(key=lambda k: k[1])

# Step 2: Group into rows
rows_detected = []
current_row = []
row_threshold = 20  # max y-difference to consider same row

for i, box in enumerate(key_boxes):
    if not current_row:
        current_row.append(box)
        continue
    _, cy, _, _, _, _ = box
    _, prev_cy, _, _, _, _ = current_row[-1]
    if abs(cy - prev_cy) <= row_threshold:
        current_row.append(box)
    else:
        rows_detected.append(current_row)
        current_row = [box]
if current_row:
    rows_detected.append(current_row)

# Step 3: Sort each row left-to-right
for row in rows_detected:
    row.sort(key=lambda k: k[0])

# --- ASSIGN QWERTY LABELS ---
key_map = {}
for row_idx, row in enumerate(rows_detected):
    if row_idx >= len(rows):
        break
    labels = rows[row_idx]
    for col_idx, box in enumerate(row):
        if col_idx >= len(labels):
            break
        label = labels[col_idx]
        cx, cy, x, y, w, h = box
        key_map[label] = (cx, cy)

        # Draw on frame
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x+5, y+h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
        cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)

# --- OUTPUT ---
print("Detected keys and their centers:")
for k, v in key_map.items():
    print(f"{k}: {v}")

np.save("keyboard_keymap.npy", key_map)

cv2.imshow("Keyboard Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
