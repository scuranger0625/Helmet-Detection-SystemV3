import cv2
import numpy as np


def compute_edge_density(region):

    # ==========================================
    # BGR -> Gray
    # ==========================================
    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY
    )

    # ==========================================
    # Canny edge
    # ==========================================
    edges = cv2.Canny(
        gray,
        80,
        160
    )

    # ==========================================
    # edge ratio
    # ==========================================
    edge_ratio = (
        edges.mean() / 255.0
    )

    return float(edge_ratio)


def estimate_head_region(
    frame,
    person_bbox,
    num_bands=4
):

    # ==========================================
    # person bbox
    # ==========================================
    x1, y1, x2, y2 = map(
        int,
        person_bbox
    )

    person_w = x2 - x1
    person_h = y2 - y1

    # bbox 異常
    if person_w <= 0 or person_h <= 0:
        return None

    # ==========================================
    # 只看 upper body
    # ==========================================
    upper_h = int(person_h * 0.38)

    upper_region = frame[
        y1:y1 + upper_h,
        x1:x2
    ]

    if upper_region.size == 0:
        return None

    # ==========================================
    # split vertical bands
    # ==========================================
    band_h = upper_region.shape[0] // num_bands

    best_score = -1
    best_band = None

    for i in range(num_bands):

        by1 = i * band_h
        by2 = (i + 1) * band_h

        band = upper_region[
            by1:by2,
            :
        ]

        if band.size == 0:
            continue

        # ==========================================
        # edge density
        # ==========================================
        edge_score = compute_edge_density(
            band
        )

        # ==========================================
        # 上方加權
        # 越上面越像頭
        # ==========================================
        position_weight = (
            1.0 - (i / num_bands)
        )

        score = (
            0.75 * edge_score +
            0.25 * position_weight
        )

        if score > best_score:

            best_score = score
            best_band = (by1, by2)

    # 找不到
    if best_band is None:
        return None

    # ==========================================
    # convert back to frame coords
    # ==========================================
    by1, by2 = best_band

    head_y1 = y1 + by1
    head_y2 = y1 + by2

    # ==========================================
    # head width
    # ==========================================
    center_x = (x1 + x2) // 2

    head_w = int(person_w * 0.55)

    hx1 = max(
        0,
        center_x - head_w // 2
    )

    hx2 = min(
        frame.shape[1],
        center_x + head_w // 2
    )

    return (
        hx1,
        head_y1,
        hx2,
        head_y2
    )


def crop_head(
    frame,
    person_bbox
):

    # ==========================================
    # estimate head region
    # ==========================================
    head_bbox = estimate_head_region(
        frame,
        person_bbox
    )

    if head_bbox is None:
        return None

    hx1, hy1, hx2, hy2 = head_bbox

    # 防止空 crop
    if hx2 <= hx1 or hy2 <= hy1:
        return None

    # ==========================================
    # crop head
    # ==========================================
    head_crop = frame[
        hy1:hy2,
        hx1:hx2
    ]

    if head_crop.size == 0:
        return None

    return head_crop, head_bbox