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
DIGITS = set("0123456789")

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
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    bs = block_size if block_size % 2 == 1 else block_size + 1
    mask_adaptive = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        bs, C
    )

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_adaptive = cv2.morphologyEx(mask_adaptive, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    mask_adaptive = cv2.erode(mask_adaptive, kernel_small, iterations=1)

    gray_bi = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray_bi, 40, 160)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    mask_edges = cv2.dilate(edges, kernel_small, iterations=1)

    mask_union = cv2.bitwise_or(mask_adaptive, mask_edges)

    return gray, mask_adaptive, mask_edges, mask_union

# --- FIND CONTOURS ---
def find_key_contours(binary_mask, img_shape):
    contours, hierarchy = cv2.findContours(binary_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img_shape[:2]
    image_area = float(W * H)
    min_area = image_area * 0.00001
    max_area = image_area * 0.08

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
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if len(approx) < 4 or len(approx) > 14:
            continue
        candidates.append(cnt)

    candidates = sorted(candidates, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    return candidates

# --- OCR HELPERS ---
def _run_tesseract(pil_img, whitelist):
    whitelist_token = shlex.quote(whitelist)
    for psm in (10, 8, 6, 7, 13):
        config = f'--oem 1 --psm {psm} -c tessedit_char_whitelist={whitelist_token}'
        text = pytesseract.image_to_string(pil_img, config=config).strip().replace("\n", "")
        if text:
            return text
    return ""

def ocr_label_from_roi(gray, bbox, valid_keys=VALID_KEYS):
    x, y, w, h = bbox
    margin = int(0.10 * min(w, h))
    rx1 = max(0, x + margin)
    ry1 = max(0, y + margin)
    rx2 = min(gray.shape[1], x + w - margin)
    ry2 = min(gray.shape[0], y + h - margin)
    roi = gray[ry1:ry2, rx1:rx2]
    if roi.size == 0 or roi.shape[0] < 6 or roi.shape[1] < 6:
        return None

    roi_big = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
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

    if len(text) == 1 and text.isalpha():
        text = text.upper()
    mapping = {'|': 'I', 'l': 'I'}  # do not map '0' to 'O'
    text = mapping.get(text, text)

    return text if text in valid_keys else None

# --- GEOMETRY HELPERS ---
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

def nms_boxes(boxes, iou_thresh=0.35):
    if not boxes:
        return []
    areas = [(w * h) for (_, _, w, h) in boxes]
    idxs = list(range(len(boxes)))
    idxs.sort(key=lambda i: areas[i], reverse=True)
    kept = []
    while idxs:
        i = idxs.pop(0)
        keep = True
        for k in kept:
            if iou_box(boxes[i], boxes[k]) > iou_thresh:
                keep = False
                break
        if keep:
            kept.append(i)
        idxs = [j for j in idxs if iou_box(boxes[j], boxes[i]) <= iou_thresh]
    return [boxes[i] for i in kept]

def remove_nested_boxes(boxes, pad=2, area_ratio_thresh=0.85):
    if not boxes:
        return []
    areas = [w * h for (x, y, w, h) in boxes]
    idxs = list(range(len(boxes)))
    idxs.sort(key=lambda i: areas[i], reverse=True)

    def contains(outer, inner):
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return (ix >= ox + pad and iy >= oy + pad and
                ix + iw <= ox + ow - pad and iy + ih <= oy + oh - pad)

    keep = [True] * len(boxes)
    for i_pos, i in enumerate(idxs):
        if not keep[i]:
            continue
        oi = boxes[i]
        ai = areas[i]
        for j in idxs[i_pos + 1:]:
            if not keep[j]:
                continue
            oj = boxes[j]
            aj = areas[j]
            if contains(oi, oj) and (aj / ai) <= area_ratio_thresh:
                keep[j] = False
    return [boxes[k] for k in range(len(boxes)) if keep[k]]

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

# --- BUILD DETECTED ROWS ---
def build_detected_rows(results):
    idxs = [i for i, r in enumerate(results) if r.get("bbox")]
    if not idxs:
        return [], {}, []
    heights = [results[i]["bbox"]["h"] for i in idxs]
    typical_h = statistics.median(heights) if heights else 50
    centers = [(results[i]["center"]["cx"], results[i]["center"]["cy"]) for i in idxs]
    row_positions = cluster_rows_by_y(centers, typical_h)
    rows = []
    for pos_cluster in row_positions:
        row_indices = [idxs[p] for p in pos_cluster]
        row_sorted = sorted(row_indices, key=lambda ridx: results[ridx]["center"]["cx"])
        rows.append(row_sorted)
    index_to_row_info = {}
    for ridx_row, row in enumerate(rows):
        for cpos, res_idx in enumerate(row):
            index_to_row_info[res_idx] = (ridx_row, cpos)
    return rows, index_to_row_info, row_positions

# --- MAP DETECTED ROWS TO QWERTY LAYOUT ROWS ---
def map_rows_to_layout(results, qwerty_layout):
    rows, _, _ = build_detected_rows(results)
    if not rows:
        return {}
    det_row_labels = [set(results[i]["label"] for i in row if results[i]["label"] and results[i]["label"] != "?") for row in rows]
    layout_row_sets = [set(row) for row in qwerty_layout]
    scores = [[len(det_row_labels[dr] & layout_row_sets[lr]) for lr in range(len(qwerty_layout))] for dr in range(len(rows))]
    row_map = {}
    used_layout = set()
    for dr in range(len(rows)):
        lr_best = max(range(len(qwerty_layout)), key=lambda lr: (scores[dr][lr], lr not in used_layout))
        row_map[dr] = lr_best
        used_layout.add(lr_best)
    return row_map

# --- QWERTY CORRECTION ---
def correct_labels_by_qwerty(results, qwerty_layout, min_row_len=5, match_thresh=0.6):
    idxs = [i for i, r in enumerate(results) if r.get("bbox")]
    if not idxs:
        return results
    heights = [results[i]["bbox"]["h"] for i in idxs]
    typical_h = statistics.median(heights) if heights else 50
    centers_coords = [(results[i]["center"]["cx"], results[i]["center"]["cy"]) for i in idxs]
    row_positions = cluster_rows_by_y(centers_coords, typical_h)
    for pos_cluster in row_positions:
        row_indices = [idxs[p] for p in pos_cluster]
        row_keys = [results[i] for i in row_indices]
        row_pairs = sorted(list(zip(row_indices, row_keys)), key=lambda rk: rk[1]["center"]["cx"])
        row_indices_sorted = [p[0] for p in row_pairs]
        row_keys_sorted = [p[1] for p in row_pairs]
        labels = [r["label"] for r in row_keys_sorted]
        if len(labels) < min_row_len:
            continue
        best = {"score": -1.0, "layout": None, "offset": None, "overlap_start_det": None, "overlap_start_lay": None, "overlap_len": 0}
        for layout_row in qwerty_layout:
            Ld = len(labels)
            Ll = len(layout_row)
            for offset in range(-Ld + 1, Ll):
                start_det = max(0, -offset)
                start_lay = max(0, offset)
                overlap = min(Ld - start_det, Ll - start_lay)
                if overlap <= 0:
                    continue
                match_count = 0
                for i in range(overlap):
                    if labels[start_det + i] == layout_row[start_lay + i]:
                        match_count += 1
                match_ratio = match_count / float(overlap)
                if match_ratio > best["score"] or (math.isclose(match_ratio, best["score"]) and overlap > best["overlap_len"]):
                    best.update(score=match_ratio, layout=layout_row, offset=offset,
                                overlap_start_det=start_det, overlap_start_lay=start_lay, overlap_len=overlap)
        if best["layout"] is not None and best["score"] >= match_thresh and best["overlap_len"] >= min_row_len:
            for i in range(best["overlap_len"]):
                det_idx_in_row = best["overlap_start_det"] + i
                lay_idx = best["overlap_start_lay"] + i
                row_keys_sorted[det_idx_in_row]["label"] = best["layout"][lay_idx]
            for i, res_idx in enumerate(row_indices_sorted):
                results[res_idx] = row_keys_sorted[i]
    return results

# --- ROW-AWARE AMBIGUITY FIX (convert number-row 'O' to '0') ---
def fix_ambiguous_by_row(results):
    rows, _, _ = build_detected_rows(results)
    if not rows:
        return results
    if len(rows) >= 2:
        number_row = rows[1]
    else:
        number_row = rows[0]
    for res_idx in number_row:
        lbl = results[res_idx]["label"]
        if lbl == 'O':
            results[res_idx]["label"] = '0'
    return results

# --- DUPLICATE LABEL RESOLUTION (demote losers to '?', prefer expected row; for digits, prefer top-most) ---
def resolve_duplicate_labels(results, qwerty_layout):
    label_pos = {lbl: (ri, ci) for ri, row in enumerate(qwerty_layout) for ci, lbl in enumerate(row)}
    idxs = [i for i, r in enumerate(results) if r.get("bbox") and r.get("label") and r["label"] != "?"]
    if not idxs:
        return results

    rows, index_to_row_info, _ = build_detected_rows(results)
    row_map = map_rows_to_layout(results, qwerty_layout)

    widths = [results[i]["bbox"]["w"] for i in idxs]
    typical_w = statistics.median(widths) if widths else 30

    def expected_neighbors(lbl):
        if lbl not in label_pos:
            return {}
        r, c = label_pos[lbl]
        exp = {}
        row = qwerty_layout[r]
        if c - 1 >= 0:
            exp["left"] = row[c - 1]
        if c + 1 < len(row):
            exp["right"] = row[c + 1]
        if r - 1 >= 0 and c < len(qwerty_layout[r - 1]):
            exp["above"] = qwerty_layout[r - 1][c]
        if r + 1 < len(qwerty_layout) and c < len(qwerty_layout[r + 1]):
            exp["below"] = qwerty_layout[r + 1][c]
        return exp

    def neighbor_score(res_idx, lbl):
        score = 0
        exp = expected_neighbors(lbl)
        row_id, col_pos = index_to_row_info.get(res_idx, (None, None))
        if row_id is None:
            return score
        row = rows[row_id]
        if col_pos - 1 >= 0 and exp.get("left"):
            left_idx = row[col_pos - 1]
            if results[left_idx]["label"] == exp["left"]:
                score += 1
        if col_pos + 1 < len(row) and exp.get("right"):
            right_idx = row[col_pos + 1]
            if results[right_idx]["label"] == exp["right"]:
                score += 1
        cx = results[res_idx]["center"]["cx"]
        x_thresh = typical_w * 1.6
        if exp.get("above") and row_id - 1 >= 0:
            above_row = rows[row_id - 1]
            if above_row:
                nearest = min(above_row, key=lambda j: abs(results[j]["center"]["cx"] - cx))
                if abs(results[nearest]["center"]["cx"] - cx) <= x_thresh and results[nearest]["label"] == exp["above"]:
                    score += 1
        if exp.get("below") and row_id + 1 < len(rows):
            below_row = rows[row_id + 1]
            if below_row:
                nearest = min(below_row, key=lambda j: abs(results[j]["center"]["cx"] - cx))
                if abs(results[nearest]["center"]["cx"] - cx) <= x_thresh and results[nearest]["label"] == exp["below"]:
                    score += 1
        return score

    label_groups = {}
    for i in idxs:
        lbl = results[i]["label"]
        label_groups.setdefault(lbl, []).append(i)

    for lbl, inds in label_groups.items():
        if len(inds) <= 1:
            continue

        exp_layout_row = label_pos.get(lbl, (None, None))[0]
        in_expected_rows = []
        out_rows = []
        for j in inds:
            det_row = index_to_row_info.get(j, (None, None))[0]
            mapped = row_map.get(det_row, None)
            if mapped == exp_layout_row:
                in_expected_rows.append(j)
            else:
                out_rows.append(j)
        candidates = in_expected_rows if in_expected_rows else inds

        scored = []
        for j in candidates:
            s = neighbor_score(j, lbl)
            area = results[j]["bbox"]["w"] * results[j]["bbox"]["h"]
            cy = results[j]["center"]["cy"]
            scored.append((s, area, cy, j))

        if lbl in DIGITS:
            scored.sort(key=lambda t: (t[0], -t[2], t[1]), reverse=True)
        else:
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        keep_id = scored[0][3]

        for _, _, _, j in scored[1:]:
            if j != keep_id:
                results[j]["inferred"] = True
                results[j]["label"] = "?"
        for j in out_rows:
            if j != keep_id:
                results[j]["inferred"] = True
                results[j]["label"] = "?"

    return results

# --- FILL MISSING KEYS BY LAYOUT GAPS (updated per-row version) ---
def fill_missing_by_layout(results, qwerty_layout):
    rows, _, _ = build_detected_rows(results)
    if not rows:
        return results
    row_map = map_rows_to_layout(results, qwerty_layout)

    def assign_between_row(row_indices, left_label, right_label, target_label):
        left_idx = next((i for i in row_indices if results[i]["label"] == left_label), None)
        right_idx = next((i for i in row_indices if results[i]["label"] == right_label), None)
        if left_idx is None or right_idx is None:
            return False
        cx_left = results[left_idx]["center"]["cx"]
        cx_right = results[right_idx]["center"]["cx"]
        if cx_left >= cx_right:
            return False
        between = [i for i in row_indices
                   if results[i]["label"] == "?"
                   and cx_left < results[i]["center"]["cx"] < cx_right]
        if not between:
            return False
        mid = (cx_left + cx_right) / 2.0
        chosen = min(between, key=lambda i: abs(results[i]["center"]["cx"] - mid))
        results[chosen]["label"] = target_label
        results[chosen]["inferred"] = True
        return True

    for det_row_id, row_indices in enumerate(rows):
        layout_row_id = row_map.get(det_row_id, None)
        if layout_row_id is None:
            continue
        layout_row = qwerty_layout[layout_row_id]
        labels_in_row = {results[idx]["label"] for idx in row_indices if results[idx]["label"] != "?"}
        for ci, lbl in enumerate(layout_row):
            if lbl in labels_in_row:
                continue
            left_lbl = layout_row[ci - 1] if ci - 1 >= 0 else None
            right_lbl = layout_row[ci + 1] if ci + 1 < len(layout_row) else None
            if left_lbl and right_lbl:
                assign_between_row(row_indices, left_lbl, right_lbl, lbl)
    return results

# --- FINAL NEIGHBOR-ONLY GAP FILL (no row mapping) ---
def fill_missing_by_neighbors_no_map(results, qwerty_layout):
    rows, _, _ = build_detected_rows(results)
    if not rows:
        return results

    triplets = []
    for layout_row in qwerty_layout:
        for i in range(1, len(layout_row) - 1):
            left_lbl = layout_row[i - 1]
            target_lbl = layout_row[i]
            right_lbl = layout_row[i + 1]
            triplets.append((left_lbl, target_lbl, right_lbl))

    for row_indices in rows:
        labels_in_row = {results[idx]["label"]: idx for idx in row_indices if results[idx]["label"] != "?"}
        for left_lbl, target_lbl, right_lbl in triplets:
            if left_lbl in labels_in_row and right_lbl in labels_in_row and target_lbl not in labels_in_row:
                left_idx = labels_in_row[left_lbl]
                right_idx = labels_in_row[right_lbl]
                cx_left = results[left_idx]["center"]["cx"]
                cx_right = results[right_idx]["center"]["cx"]
                if cx_left >= cx_right:
                    continue
                between = [i for i in row_indices
                           if results[i]["label"] == "?"
                           and cx_left < results[i]["center"]["cx"] < cx_right]
                if not between:
                    continue
                mid = (cx_left + cx_right) / 2.0
                chosen = min(between, key=lambda i: abs(results[i]["center"]["cx"] - mid))
                results[chosen]["label"] = target_lbl
                results[chosen]["inferred"] = True

    return results

# --- SPACEBAR LABELING BASED ON 'B' KEY POSITION AND WIDTH ---
def label_spacebar_by_B(results, width_factor=3.0, dx_thresh_factor=1.0, dy_thresh_factor=0.6):
    B_key = next((r for r in results if r.get("bbox") and r.get("label") == "B"), None)
    if B_key is None:
        for alt in ("V", "N"):
            B_key = next((r for r in results if r.get("bbox") and r.get("label") == alt), None)
            if B_key:
                break

    letter_widths = [r["bbox"]["w"] for r in results if r.get("bbox") and r.get("label") and len(r["label"]) == 1 and r["label"].isalpha()]
    typical_w = statistics.median(letter_widths) if letter_widths else statistics.median([r["bbox"]["w"] for r in results if r.get("bbox")]) if results else 40

    cx_ref = B_key["center"]["cx"] if B_key else None
    cy_ref = B_key["center"]["cy"] if B_key else None
    h_ref = B_key["bbox"]["h"] if B_key else statistics.median([r["bbox"]["h"] for r in results if r.get("bbox")]) if results else 30
    w_ref = B_key["bbox"]["w"] if B_key else typical_w

    expected_cx = cx_ref if cx_ref is not None else statistics.median([r["center"]["cx"] for r in results if r.get("bbox")]) if results else None
    expected_cy = (cy_ref + h_ref) if cy_ref is not None else None
    if expected_cx is None or expected_cy is None:
        return results

    dx_thresh = dx_thresh_factor * w_ref
    dy_thresh = dy_thresh_factor * h_ref

    idxs = [i for i, r in enumerate(results) if r.get("bbox")]
    if not idxs:
        return results
    heights = [results[i]["bbox"]["h"] for i in idxs]
    typical_h = statistics.median(heights) if heights else 50
    centers = [(results[i]["center"]["cx"], results[i]["center"]["cy"]) for i in idxs]
    row_positions = cluster_rows_by_y(centers, typical_h)
    if not row_positions:
        return results
    bottom_cluster = row_positions[-1]
    bottom_row_indices = [idxs[p] for p in bottom_cluster]

    candidates = []
    for i in bottom_row_indices:
        r = results[i]
        w = r["bbox"]["w"]
        cx = r["center"]["cx"]
        cy = r["center"]["cy"]
        if w >= width_factor * w_ref and abs(cx - expected_cx) <= dx_thresh and abs(cy - expected_cy) <= dy_thresh:
            candidates.append((w, i))

    if not candidates:
        return results

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, best_idx = candidates[0]
    results[best_idx]["label"] = "SPACE"
    results[best_idx]["inferred"] = False
    return results

# --- SPACEBAR HEURISTICS (optional center marker) ---
def get_space_center(results, offset_factor=1.25):
    reference_key = None
    for key in ["B", "V", "N"]:
        for r in results:
            if r.get("bbox") and r["label"] == key:
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
    boxes = [r for r in results if r.get("bbox")]
    widths = [r["bbox"]["w"] for r in boxes]
    heights = [r["bbox"]["h"] for r in boxes]
    if not widths or not heights:
        sc = get_space_center(results)
        if sc:
            return {"label": "SPACE", "bbox": None, "center": sc, "inferred": True}
        return None
    typical_h = statistics.median(heights)
    centers = [(r["center"]["cx"], r["center"]["cy"]) for r in boxes]
    row_indices = cluster_rows_by_y(centers, typical_h)
    if not row_indices:
        sc = get_space_center(results)
        if sc:
            return {"label": "SPACE", "bbox": None, "center": sc, "inferred": True}
        return None
    filtered_idxs = [i for i, r in enumerate(results) if r.get("bbox")]
    bottom_cluster = row_indices[-1]
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

    contour_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= 12 and h >= 12:
            contour_boxes.append((x, y, w, h))

    all_boxes = nms_boxes(contour_boxes, iou_thresh=0.3)
    all_boxes = remove_nested_boxes(all_boxes, pad=2, area_ratio_thresh=0.85)

    raw_results = []
    for (x, y, w, h) in all_boxes:
        label = ocr_label_from_roi(gray, (x, y, w, h), VALID_KEYS)
        cx, cy = int(x + w / 2), int(y + h / 2)
        raw_results.append({
            "label": label if label else "?",
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "center": {"cx": cx, "cy": cy},
            "inferred": False if label else True
        })
        if verbose:
            print(f"Box @ ({x},{y},{w},{h}) -> {label if label else '?'}")

    qwerty_layout = [
        ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
    ]

    if any(r["label"] != "?" for r in raw_results):
        raw_results = correct_labels_by_qwerty(raw_results, qwerty_layout, min_row_len=5, match_thresh=0.6)

    raw_results = fix_ambiguous_by_row(raw_results)
    dedup_results = resolve_duplicate_labels(raw_results, qwerty_layout)
    filled_results = fill_missing_by_layout(dedup_results, qwerty_layout)
    filled_results = fill_missing_by_neighbors_no_map(filled_results, qwerty_layout)
    final_results = label_spacebar_by_B(filled_results, width_factor=3.0, dx_thresh_factor=1.0, dy_thresh_factor=0.6)

    annotated = img.copy()
    for r in final_results:
        if not r.get("bbox"):
            continue
        x, y, w, h = r["bbox"]["x"], r["bbox"]["y"], r["bbox"]["w"], r["bbox"]["h"]
        lbl = r["label"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, lbl, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if lbl != "?" else (0, 165, 255), 1)

    space_info = detect_spacebar(final_results)
    if space_info:
        cx, cy = space_info["center"]["cx"], space_info["center"]["cy"]
        cv2.circle(annotated, (cx, cy), 5, (255, 0, 0), -1)

    return annotated, final_results

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
