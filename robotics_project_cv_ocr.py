import cv2
import pytesseract
import numpy as np
import os
import json
from PIL import Image

# OCR based keyboard key detection
# TODO: Clean up the recognition, maybe give concrete list for the OCR detection

# --- CONFIGURATION ---

# If tesseract isn't on PATH, uncomment and adjust this path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_CONFIG = r'--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

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
    """
    Preprocess image to get clearer separation between individual keys.
    Uses adaptive thresholding instead of just Canny.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold helps separate individual keys under varied lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 8
    )

    # Morphological operations to clean noise but keep key gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Lightly erode to widen the gaps between keys
    closed = cv2.erode(closed, kernel, iterations=1)

    return closed, gray


# --- CONTOUR DETECTION ---

def find_key_contours(edges):
    """
    Finds smaller contours representing individual keys.
    Uses hierarchy to include nested contours if any.
    """
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    key_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150 or area > 20000:  # ignore too small or too large areas
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)
        if 0.5 < aspect < 3.5 and 20 < w < 200 and 20 < h < 200:
            key_contours.append(cnt)

    # Sort left-to-right, top-to-bottom for consistent order
    key_contours = sorted(key_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    return key_contours


# --- OCR PER KEY ---

def ocr_label_from_roi(gray, bbox):
    x, y, w, h = bbox
    roi = gray[y:y+h, x:x+w]

    # Resize for better OCR
    roi = cv2.resize(roi, (w * 2, h * 2))
    roi = cv2.bitwise_not(roi)  # invert to make text dark on light
    pil = Image.fromarray(roi)

    text = pytesseract.image_to_string(pil, config=OCR_CONFIG)
    text = text.strip()
    if len(text) > 3:  # skip junk
        text = text[:3]
    return text


# --- MAIN DETECTION PIPELINE ---

def detect_keys(image_path, verbose=False):
    img = load_and_resize(image_path)
    edges, gray = preprocess_for_contours(img)
    contours = find_key_contours(edges)

    annotated = img.copy()
    results = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        label = ocr_label_from_roi(gray, (x, y, w, h))
        if not label:
            continue

        # Draw bounding box and label
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        result = {
            "label": label,
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "center": {"cx": int(x + w/2), "cy": int(y + h/2)}
        }
        results.append(result)
        if verbose:
            print(f"Found: {result}")

    return annotated, results


# --- MAIN ENTRY POINT ---

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "keyboard_image.jpg")

    if not os.path.exists(image_path):
        print(f"Error: keyboard.jpg not found in {script_dir}")
        return

    print(f"Processing image: {image_path}")
    annotated, results = detect_keys(image_path, verbose=True)

    out_path = os.path.join(script_dir, "annotated_keyboard.jpg")
    cv2.imwrite(out_path, annotated)

    print(f"\nAnnotated image saved to: {out_path}")
    print("\nDetected keys (JSON):")
    print(json.dumps(results, indent=2))

    # Optional display window
    cv2.imshow("Detected Keys", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
