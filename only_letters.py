import cv2
import pytesseract
import numpy as np
import os
import json
from PIL import Image
import math

# --- CONFIGURATION ---

OCR_CONFIG = r'--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# --- IMAGE LOADING ---

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
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 8
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    closed = cv2.erode(closed, kernel, iterations=1)

    return closed, gray


# --- CONTOUR DETECTION ---

def find_key_contours(edges):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    key_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150 or area > 20000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)
        if 0.5 < aspect < 3.5 and 20 < w < 200 and 20 < h < 200:
            key_contours.append(cnt)

    key_contours = sorted(key_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    return key_contours


# --- OCR PER KEY ---

def ocr_label_from_roi(gray, bbox):
    x, y, w, h = bbox
    roi = gray[y:y+h, x:x+w]

    roi = cv2.resize(roi, (w * 2, h * 2))
    roi = cv2.bitwise_not(roi)
    pil = Image.fromarray(roi)

    text = pytesseract.image_to_string(pil, config=OCR_CONFIG)
    text = text.strip()

    if len(text) == 1 and text.isalpha():
        text = text.upper()

    return text if text.isalpha() else None


# --- SAFEGUARD FOR DUPLICATES ---

def is_duplicate(new_bbox, detected_bboxes, min_distance=50):
    new_cx, new_cy = new_bbox[0] + new_bbox[2] // 2, new_bbox[1] + new_bbox[3] // 2
    for (cx, cy, w, h) in detected_bboxes:
        dist = math.sqrt((new_cx - cx) ** 2 + (new_cy - cy) ** 2)
        if dist < min_distance:
            return True
    return False


# --- HANDLE LABEL DUPLICATION --- 

def is_unique_label(new_label, detected_labels):
    if new_label in detected_labels:
        return False
    return True


# --- CONTEXTUAL INFERENCE BASED ON NEIGHBORING KEYS ---

def infer_missing_keys(detected_keys, detected_labels):
    """
    Try to infer missing keys based on detected neighbors.
    """
    inferred_keys = []
    qwerty_layout = [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
    ]
    
    # We will infer missing keys based on the QWERTY layout
    for row in qwerty_layout:
        for key in row:
            if key not in detected_labels:
                # We can infer this key based on neighbors
                neighbors = get_neighbors(key, row)
                inferred_keys.append({'label': key, 'neighbors': neighbors})

    return inferred_keys


def get_neighbors(key, row):
    """Returns expected neighboring keys in the QWERTY layout for alphabetic keys."""
    idx = row.index(key)
    neighbors = []
    if idx > 0:
        neighbors.append(row[idx - 1])  # Left
    if idx < len(row) - 1:
        neighbors.append(row[idx + 1])  # Right
    return neighbors


# --- MAIN DETECTION PIPELINE ---

def detect_keys(image_path):
    img = load_and_resize(image_path)
    edges, gray = preprocess_for_contours(img)
    contours = find_key_contours(edges)

    annotated = img.copy()
    results = []
    detected_bboxes = []
    detected_labels = []  
    detected_keys = []  

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        label = ocr_label_from_roi(gray, (x, y, w, h))
        if not label:
            continue

        new_bbox = (x, y, w, h)

        # Skip duplicate keys (based on bounding box proximity)
        if is_duplicate(new_bbox, detected_bboxes, min_distance=50):
            continue

        # Ensure label uniqueness
        if not is_unique_label(label, detected_labels):
            continue

        # Mark this bounding box as detected
        detected_bboxes.append(new_bbox)
        detected_labels.append(label)
        detected_keys.append({'label': label, 'bbox': new_bbox})

        # Draw bounding box and label
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        result = {
            "label": label,
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "center": {"cx": int(x + w / 2), "cy": int(y + h / 2)}
        }
        results.append(result)

    # Now try to infer missing keys based on neighboring context
    inferred_keys = infer_missing_keys(detected_keys, detected_labels)

    return annotated, results + inferred_keys


# --- MAIN ENTRY POINT ---

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "keyboard_image.jpg")

    if not os.path.exists(image_path):
        print(f"Error: keyboard.jpg not found in {script_dir}")
        return

    print(f"Processing image: {image_path}")

    annotated_img, results = detect_keys(image_path)

    if annotated_img is None:
        print("No valid key layout detected.")
        return

    out_path = os.path.join(script_dir, "annotated_keyboard_image.jpg")
    cv2.imwrite(out_path, annotated_img)

    print(f"\nAnnotated image saved to: {out_path}")
    print("\nDetected keys (JSON):")
    print(json.dumps(results, indent=2))

    # Optional display window
    cv2.imshow("Detected Keys", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
