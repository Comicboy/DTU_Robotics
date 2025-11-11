import cv2
import pytesseract
import numpy as np
import os
import json
import math
import statistics
from PIL import Image

# --- CONFIGURATION ---
OCR_CONFIG = (
    r'--psm 10 '
    r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789`~!@#$%^&*()-_=+[]{}\\|;:\'",.<>?'
)

# --- IMAGE HANDLING ---
def load_and_resize(path, max_w=1280):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Couldn't read image: {path}")
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# --- PREPROCESSING ---
def preprocess_for_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        21, 8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    closed = cv2.erode(closed, kernel, iterations=1)
    return closed, gray

# --- FIND CONTOURS ---
def find_key_contours(edges):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    key_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150 or area > 20000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        if 0.5 < aspect < 3.5 and 20 < w < 200 and 20 < h < 200:
            key_contours.append(cnt)
    key_contours = sorted(key_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    return key_contours

# --- OCR EXTRACTION ---
def ocr_label_from_roi(gray, bbox, valid_keys):
    x, y, w, h = bbox
    pad = int(min(w, h) * 0.1)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = gray[y1:y2, x1:x2]
    roi = cv2.resize(roi, (w * 2, h * 2))
    roi = cv2.bitwise_not(roi)
    pil = Image.fromarray(roi)
    text = pytesseract.image_to_string(pil, config=OCR_CONFIG).strip().replace("\n", "")
    if len(text) == 1 and text.isalpha():
        text = text.upper()
    if text in valid_keys:
        return text
    return None

# --- DUPLICATE FILTERING ---
def is_duplicate(new_bbox, detected_bboxes, min_distance=40):
    x, y, w, h = new_bbox
    cx, cy = x + w / 2, y + h / 2
    for (ox, oy, ow, oh) in detected_bboxes:
        ocx, ocy = ox + ow / 2, oy + oh / 2
        if math.hypot(cx - ocx, cy - ocy) < min_distance:
            return True
    return False

# --- UNIQUE LABEL CHECK ---
def is_unique_label(label, detected_labels):
    return label not in detected_labels

# --- CLUSTER ROWS BY Y ---
def cluster_rows_by_y(centers, height_est):
    if not centers:
        return []
    ys = [(i, c[1]) for i, c in enumerate(centers)]
    ys_sorted = sorted(ys, key=lambda t: t[1])
    clusters = []
    current = [ys_sorted[0][0]]
    prev_y = ys_sorted[0][1]
    for idx, y in ys_sorted[1:]:
        if abs(y - prev_y) <= max(1.2 * height_est, 12):
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
        prev_y = y
    clusters.append(current)
    return clusters

# --- LABEL CORRECTION BY QWERTY ---
def correct_labels_by_qwerty(results, qwerty_layout, min_seq=5, match_thresh=0.6):
    key_heights = [r["bbox"]["h"] for r in results]
    typical_h = statistics.median(key_heights) if key_heights else 50
    centers_coords = [(r["center"]["cx"], r["center"]["cy"]) for r in results]
    row_indices = cluster_rows_by_y(centers_coords, typical_h)

    for cluster in row_indices:
        row_keys = [results[i] for i in cluster]
        row_keys.sort(key=lambda r: r["center"]["cx"])
        labels = [r["label"] for r in row_keys]

        # Sliding sequence correction
        for start in range(len(labels) - min_seq + 1):
            seq_labels = labels[start:start + min_seq]
            best_row = None
            best_match = 0
            for layout_row in qwerty_layout:
                match_count = sum(1 for a, b in zip(seq_labels, layout_row[start:start + min_seq]) if a == b)
                match_ratio = match_count / min_seq
                if match_ratio > best_match:
                    best_match = match_ratio
                    best_row = layout_row
            if best_match >= match_thresh:
                for i, lbl in enumerate(seq_labels):
                    correct_label = best_row[start + i]
                    row_keys[start + i]["label"] = correct_label
        # Update results
        for i, idx in enumerate(cluster):
            results[idx] = row_keys[i]
    return results

# --- SPACE KEY CENTER BASED ON V/B/N ---
def get_space_center(results, offset_factor=1.25):
    reference_key = None
    for key in ["B", "V", "N"]:  # prefer B if available
        for r in results:
            if r["label"] == key:
                reference_key = r
                break
        if reference_key:
            break
    if not reference_key:
        return None
    cx_ref = reference_key["center"]["cx"]
    cy_ref = reference_key["center"]["cy"]
    h_ref = reference_key["bbox"]["h"]
    space_center = {"cx": cx_ref, "cy": int(cy_ref + offset_factor * h_ref)}
    return space_center

# --- MAIN DETECTION PIPELINE ---
def detect_keyboard_keys(image_path, verbose=False):
    img = load_and_resize(image_path)
    edges, gray = preprocess_for_contours(img)
    contours = find_key_contours(edges)

    annotated = img.copy()
    results = []
    detected_bboxes = []
    detected_labels = []
    key_widths, key_heights = [], []
    centers = []

    valid_keys = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()-_=+[]{}\\|;:'\",.<>?/")

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        label = ocr_label_from_roi(gray, (x, y, w, h), valid_keys)
        if not label:
            continue
        if is_duplicate((x, y, w, h), detected_bboxes):
            continue
        if not is_unique_label(label, detected_labels):
            continue

        detected_bboxes.append((x, y, w, h))
        detected_labels.append(label)
        key_widths.append(w)
        key_heights.append(h)
        cx, cy = int(x + w / 2), int(y + h / 2)
        centers.append((cx, cy))

        results.append({
            "label": label,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "center": {"cx": cx, "cy": cy},
            "inferred": False
        })

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        if verbose:
            print(f"Detected {label} @ ({x},{y})")

    # QWERTY layout for label correction
    qwerty_layout = [
        ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
    ]

    results = correct_labels_by_qwerty(results, qwerty_layout, min_seq=5, match_thresh=0.6)

    # Spacebar center placement
    space_center = get_space_center(results)
    if space_center:
        print(f"Space key center: {space_center}")
        cv2.circle(annotated, (space_center["cx"], space_center["cy"]), 5, (255, 0, 0), -1)

    return annotated, results

# --- MAIN ENTRY POINT ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "keyboard_image.jpg")

    if not os.path.exists(image_path):
        print(f"Error: keyboard_image.jpg not found in {script_dir}")
        return

    print(f"Processing: {image_path}")
    annotated, results = detect_keyboard_keys(image_path, verbose=True)

    out_path = os.path.join(script_dir, "annotated_keyboard_image.jpg")
    cv2.imwrite(out_path, annotated)

    print(f"\nAnnotated image saved to: {out_path}")
    print("\nDetected keys (JSON):")
    print(json.dumps(results, indent=2))

    cv2.imshow("Detected Keys", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
