import cv2
import pytesseract
import numpy as np
import os
import json
import math
import statistics
import shlex
from PIL import Image
from google.colab.patches import cv2_imshow

# --- CONFIGURATION ---
VALID_KEY_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789`~!@#$%^&*()-_=+[]{}\\|;:'\",.<>/?")
VALID_KEYS = set(VALID_KEY_CHARS)

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

# --- PREPROCESSING: build adaptive mask and edge mask, then union ---
def build_masks(img, block_size=21, C=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Light smoothing to reduce noise but keep edges
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (invert so dark regions are white)
    bs = block_size if block_size % 2 == 1 else block_size + 1
    mask_adaptive = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        bs, C
    )

    # Small morphology to avoid merging neighboring keys
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_adaptive = cv2.morphologyEx(mask_adaptive, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    mask_adaptive = cv2.erode(mask_adaptive, kernel_small, iterations=1)

    # Edge-based mask
    gray_bi = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray_bi, 40, 160)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    mask_edges = cv2.dilate(edges, kernel_small, iterations=1)

    # Union of both masks
    mask_union = cv2.bitwise_or(mask_adaptive, mask_edges)

    return gray, mask_adaptive, mask_edges, mask_union

# --- FIND CONTOURS ---
def find_key_contours(binary_mask, img_shape):
    contours, hierarchy = cv2.findContours(binary_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img_shape[:2]

    # Scale-aware thresholds
    image_area = float(W * H)
    min_area = image_area * 0.00001   # permissive
    max_area = image_area * 0.08      # allow larger keys/spacebar

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        aspect = w / float(h)
        if not (0.35 < aspect < 9.0):
            continue

        rect_area = w * h
        if rect_area <= 0:
            continue
        fill_ratio = area / float(rect_area)
        if fill_ratio < 0.20:
            continue

        # Prefer near-rectangular contours
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if len(approx) < 4 or len(approx) > 14:
            continue

        candidates.append(cnt)

    # Sort by row (y) then column (x)
    candidates = sorted(candidates, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    return candidates

# --- OCR HELPERS ---
def _run_tesseract(pil_img, whitelist):
    whitelist_token = shlex.quote(whitelist)
    # Try multiple PSMs
    for psm in (10, 8, 6, 7, 13):
        config = f'--oem 1 --psm {psm} -c tessedit_char_whitelist={whitelist_token}'
        text = pytesseract.image_to_string(pil_img, config=config).strip().replace("\n", "")
        if text:
            return text
    return ""

def ocr_label_from_roi(gray, bbox, valid_keys=VALID_KEYS):
    x, y, w, h = bbox
    # Crop inner region; too large margin can remove the legend
    margin = int(0.10 * min(w, h))
    rx1 = max(0, x + margin)
    ry1 = max(0, y + margin)
    rx2 = min(gray.shape[1], x + w - margin)
    ry2 = min(gray.shape[0], y + h - margin)
    roi = gray[ry1:ry2, rx1:rx2]
    if roi.size == 0 or roi.shape[0] < 6 or roi.shape[1] < 6:
        return None

    # Upscale for OCR
    roi_big = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # Try both binarizations
    roi_bin = cv2.threshold(roi_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    roi_inv = cv2.bitwise_not(roi_bin)

    whitelist = ''.join(sorted(valid_keys))

    pil_inv = Image.fromarray(roi_inv)
    text = _run_tesseract(pil_inv, whitelist)
    if not text:
        pil_bin = Image.fromarray(roi_bin)
        text = _run_tesseract(pil_bin, whitelist)

    if not text:
        return None

    # Normalize confusions and casing
    if len(text) == 1 and text.isalpha():
        text = text.upper()
    mapping = {'|': 'I', 'l': 'I', '0': 'O'}
    text = mapping.get(text, text)

    return text if text in valid_keys else None

# --- DUPLICATE FILTERING ---
def iou_box(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    denom = (aw * ah) + (bw * bh) - inter
    return inter / denom if denom > 0 else 0.0

def is_duplicate(new_bbox, detected_bboxes, iou_thresh=0.2, center_thresh=18):
    nx, ny, nw, nh = new_bbox
    ncx, ncy = nx + nw / 2.0, ny + nh / 2.0
    for (ox, oy, ow, oh) in detected_bboxes:
        if iou_box(new_bbox, (ox, oy, ow, oh)) > iou_thresh:
            return True
        ocx, ocy = ox + ow / 2.0, oy + oh / 2.0
        if math.hypot(ncx - ocx, ncy - ocy) < center_thresh:
            return True
    return False

def is_unique_label(label, detected_labels):
    return True  # allow duplicates; rely on geometry

# --- ROW CLUSTERING ---
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

# --- QWERTY CORRECTION (safe overlap + correct indexing) ---
def correct_labels_by_qwerty(results, qwerty_layout, min_row_len=5, match_thresh=0.6):
    # Work only with entries that have bboxes (skip inferred space etc.)
    idxs = [i for i, r in enumerate(results) if r.get("bbox")]
    if not idxs:
        return results

    heights = [results[i]["bbox"]["h"] for i in idxs]
    typical_h = statistics.median(heights) if heights else 50

    centers_coords = [(results[i]["center"]["cx"], results[i]["center"]["cy"]) for i in idxs]
    row_positions = cluster_rows_by_y(centers_coords, typical_h)

    for pos_cluster in row_positions:
        row_indices = [idxs[p] for p in pos_cluster]  # map back to original results indices
        row_keys = [results[i] for i in row_indices]
        # Sort row_keys by x and keep mapping back to indices
        row_pairs = sorted(list(zip(row_indices, row_keys)), key=lambda rk: rk[1]["center"]["cx"])
        row_indices_sorted = [p[0] for p in row_pairs]
        row_keys_sorted = [p[1] for p in row_pairs]
        labels = [r["label"] for r in row_keys_sorted]

        if len(labels) < min_row_len:
            continue

        best = {
            "score": -1.0,
            "layout": None,
            "offset": None,
            "overlap_start_det": None,
            "overlap_start_lay": None,
            "overlap_len": 0
        }

        for layout_row in qwerty_layout:
            Ld = len(labels)
            Ll = len(layout_row)
            # offset = lay_start - det_start; allow negative offsets
            for offset in range(-Ld + 1, Ll):  # safe range
                start_det = max(0, -offset)
                start_lay = max(0, offset)
                overlap = min(Ld - start_det, Ll - start_lay)
                if overlap <= 0:
                    continue
                match_count = 0
                for i in range(overlap):
                    if labels[start_det + i] == layout_row[start_lay + i]:
                        match_count += 1
                # score by ratio over overlap
                match_ratio = match_count / float(overlap)
                if match_ratio > best["score"] or (math.isclose(match_ratio, best["score"]) and overlap > best["overlap_len"]):
                    best.update(
                        score=match_ratio,
                        layout=layout_row,
                        offset=offset,
                        overlap_start_det=start_det,
                        overlap_start_lay=start_lay,
                        overlap_len=overlap
                    )

        # Apply correction only if confident and overlap is sufficiently long
        if best["layout"] is not None and best["score"] >= match_thresh and best["overlap_len"] >= min_row_len:
            for i in range(best["overlap_len"]):
                det_idx_in_row = best["overlap_start_det"] + i
                lay_idx = best["overlap_start_lay"] + i
                correct_label = best["layout"][lay_idx]
                row_keys_sorted[det_idx_in_row]["label"] = correct_label

            # Write back to original results using sorted indices mapping
            for i, res_idx in enumerate(row_indices_sorted):
                results[res_idx] = row_keys_sorted[i]

    return results

# --- SPACEBAR HEURISTICS ---
def get_space_center(results, offset_factor=1.25):
    reference_key = None
    for key in ["B", "V", "N"]:
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
    return {"cx": cx_ref, "cy": int(cy_ref + offset_factor * h_ref)}

def detect_spacebar(results):
    if not results:
        return None
    widths = [r["bbox"]["w"] for r in results if r.get("bbox")]
    heights = [r["bbox"]["h"] for r in results if r.get("bbox")]
    if not widths or not heights:
        sc = get_space_center(results)
        if sc:
            return {"label": "SPACE", "bbox": None, "center": sc, "inferred": True}
        return None

    typical_h = statistics.median(heights)
    centers = [(r["center"]["cx"], r["center"]["cy"]) for r in results if r.get("bbox")]
    row_indices = cluster_rows_by_y(centers, typical_h)
    if not row_indices:
        sc = get_space_center(results)
        if sc:
            return {"label": "SPACE", "bbox": None, "center": sc, "inferred": True}
        return None

    bottom_cluster = row_indices[-1]
    bottom_row = [results[i] for i in [j for j in range(len(results)) if results[j].get("bbox")]][:len(centers)]
    # Rebuild bottom row properly from indices mapping
    filtered_idxs = [i for i, r in enumerate(results) if r.get("bbox")]
    bottom_row = [results[filtered_idxs[k]] for k in bottom_cluster]
    bottom_row.sort(key=lambda r: r["center"]["cx"])
    med_w = statistics.median(widths)

    candidates = [r for r in bottom_row if r["bbox"]["w"] > 3.5 * med_w]
    if candidates:
        space = max(candidates, key=lambda rr: rr["bbox"]["w"])
        space["label"] = "SPACE"
        return space

    sc = get_space_center(results)
    if sc:
        return {"label": "SPACE", "bbox": None, "center": sc, "inferred": True}
    return None

# --- MAIN DETECTION PIPELINE ---
def detect_keyboard_keys(image_path, verbose=False, save_debug=True, include_unlabeled=True):
    img = load_and_resize(image_path)

    gray, mask_adaptive, mask_edges, mask_union = build_masks(img, block_size=21, C=5)
    contours = find_key_contours(mask_union, img.shape)

    # If still too few, try slightly different params
    if len(contours) < 10:
        if verbose:
            print(f"Union mask yielded {len(contours)} contours; trying alternate params...")
        gray, mask_adaptive, mask_edges, mask_union = build_masks(img, block_size=17, C=3)
        contours = find_key_contours(mask_union, img.shape)

    if verbose:
        print(f"Using {len(contours)} candidate key contours")

    if save_debug:
        cv2.imwrite("debug_mask_adaptive.png", mask_adaptive)
        cv2.imwrite("debug_mask_edges.png", mask_edges)
        cv2.imwrite("debug_mask_union.png", mask_union)

    annotated = img.copy()
    results = []
    detected_bboxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 12 or h < 12:
            continue

        if is_duplicate((x, y, w, h), detected_bboxes):
            continue
        detected_bboxes.append((x, y, w, h))

        label = ocr_label_from_roi(gray, (x, y, w, h), VALID_KEYS)
        cx, cy = int(x + w / 2), int(y + h / 2)

        if label:
            res_item = {
                "label": label,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "center": {"cx": cx, "cy": cy},
                "inferred": False
            }
            cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        else:
            if not include_unlabeled:
                continue
            res_item = {
                "label": "?",
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "center": {"cx": cx, "cy": cy},
                "inferred": True
            }
            cv2.putText(annotated, "?", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        results.append(res_item)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if verbose:
            print(f"Box @ ({x},{y},{w},{h}) -> {res_item['label']}")

    # QWERTY correction only if we have some labeled keys
    if any(r["label"] != "?" for r in results):
        qwerty_layout = [
            ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
        ]
        results = correct_labels_by_qwerty(results, qwerty_layout, min_row_len=5, match_thresh=0.6)

    # Spacebar detection (uses bbox-based entries)
    space = detect_spacebar(results)
    if space:
        if space.get("bbox"):
            bx, by, bw, bh = space["bbox"]["x"], space["bbox"]["y"], space["bbox"]["w"], space["bbox"]["h"]
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
            cv2.putText(annotated, "SPACE", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1)
        cx, cy = space["center"]["cx"], space["center"]["cy"]
        cv2.circle(annotated, (cx, cy), 5, (255, 0, 0), -1)
        if space.get("bbox") is None:
            results.append(space)
        if verbose:
            print(f"Space key: {space}")

    return annotated, results

# --- MAIN ENTRY POINT ---
def main():
    script_dir = os.getcwd()
    image_path = os.path.join(script_dir, "keyboard_image.jpg")
    if not os.path.exists(image_path):
        print(f"Error: keyboard_image.jpg not found in {script_dir}")
        return
    print(f"Processing: {image_path}")
    annotated, results = detect_keyboard_keys(image_path, verbose=True, save_debug=True, include_unlabeled=True)
    out_path = os.path.join(script_dir, "annotated_keyboard_image.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved to: {out_path}")
    print("\nDetected keys (JSON):")
    print(json.dumps(results, indent=2))
    cv2_imshow(annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
