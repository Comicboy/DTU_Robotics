import cv2
import numpy as np

# --- PARAMETERS ---
IMAGE_PATH = "keyboard_image.jpg"

# Full keyboard layout including special keys
rows_layout = [
    ["Esc","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","PrtSc","Del"],
    ["`","1","2","3","4","5","6","7","8","9","0","-","=","Backspace"],
    ["Tab","Q","W","E","R","T","Y","U","I","O","P","[","]","\\ "],
    ["CapsLock","A","S","D","F","G","H","J","K","L",";","'","Enter"],
    ["Shift_L","Z","X","C","V","B","N","M",",",".","/","Shift_R"],
    ["Ctrl_L","Win","Alt_L","Space","Alt_R","Win","Menu","Ctrl_R"]
]

# --- READ IMAGE ---
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise ValueError("Cannot read image")
frame = cv2.resize(frame, (1000, 400))

# --- PREPROCESSING ---
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 3
)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

# --- FIND CONTOURS ---
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- FILTER AND SORT KEYS ---
key_boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 8 and h > 8:  # only remove tiny noise
        key_boxes.append((x, y, w, h))

# Sort top-to-bottom by y, then left-to-right by x
key_boxes.sort(key=lambda b: (b[1], b[0]))

# --- SPLIT ROWS BASED ON LAYOUT ---
rows_detected = []
start_idx = 0
for layout_row in rows_layout:
    n_keys = len(layout_row)
    row_keys = key_boxes[start_idx:start_idx+n_keys]

    # If some rows contain large keys (like Space or Shift), find the widest boxes first
    row_keys.sort(key=lambda b: b[0])  # left-to-right
    rows_detected.append(row_keys)
    start_idx += n_keys

# --- ASSIGN LABELS ---
key_map = {}
for row_idx, row in enumerate(rows_detected):
    layout_row = rows_layout[row_idx]
    for col_idx, box in enumerate(row):
        if col_idx >= len(layout_row):
            continue
        x, y, w, h = box
        cx, cy = x + w//2, y + h//2
        key_map[layout_row[col_idx]] = (cx, cy)

        # Draw for visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, layout_row[col_idx], (x+5, y+h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)

# --- OUTPUT ---
print("Detected keys and centers:")
for k, v in key_map.items():
    print(f"{k}: {v}")

np.save("keyboard_keymap.npy", key_map)
cv2.imshow("Keyboard Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
