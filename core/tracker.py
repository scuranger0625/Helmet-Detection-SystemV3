import numpy as np


# ==========================================================
# PERSON 底部中心
# ==========================================================
def bottom_center(bbox):

    x1, y1, x2, y2 = bbox

    return np.array([

        (x1 + x2) / 2.0,

        y2

    ])


# ==========================================================
# MOTORCYCLE 騎士座位錨點
# 不用 bbox center
# 因為車子很長
# ==========================================================
def motorcycle_rider_anchor(bbox):

    x1, y1, x2, y2 = bbox

    h = y2 - y1

    return np.array([

        (x1 + x2) / 2.0,

        y1 + h * 0.35

    ])


# ==========================================================
# 計算 overlap ratio
# 用來判斷 person 是否真的坐在車上
# ==========================================================
def overlap_ratio(boxA, boxB):

    xA = max(boxA[0], boxB[0])

    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])

    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)

    inter_h = max(0, yB - yA)

    inter_area = inter_w * inter_h

    areaA = (
        (boxA[2] - boxA[0])
        * (boxA[3] - boxA[1])
    )

    if areaA <= 0:

        return 0.0

    return inter_area / areaA


# ==========================================================
# 主配對函式
# ==========================================================
def pair_person_motorcycle(

    detections,

    distance_scale=0.8,

    min_overlap=0.03

):

    persons = []

    motorcycles = []

    # ======================================================
    # 分類 detections
    # ======================================================
    for det in detections:

        cls = det["class_name"]

        if cls == "person":

            persons.append(det)

        elif cls == "motorcycle":

            motorcycles.append(det)

    pairs = []

    used_motorcycles = set()

    # ======================================================
    # 對每個 person 找最近 motorcycle
    # ======================================================
    for person in persons:

        p_bbox = person["bbox"]

        p_center = bottom_center(p_bbox)

        # ==================================================
        # 根據 person 大小動態調整距離
        # ==================================================
        person_height = (
            p_bbox[3] - p_bbox[1]
        )

        adaptive_distance = (
            person_height * distance_scale
        )

        best_idx = None

        best_dist = 1e9

        best_overlap = 0.0

        # ==================================================
        # 找最近 motorcycle
        # ==================================================
        for i, moto in enumerate(motorcycles):

            if i in used_motorcycles:

                continue

            m_bbox = moto["bbox"]

            # ==============================================
            # motorcycle rider anchor
            # ==============================================
            m_center = motorcycle_rider_anchor(
                m_bbox
            )

            # ==============================================
            # distance
            # ==============================================
            dist = np.linalg.norm(
                p_center - m_center
            )

            # ==============================================
            # overlap
            # ==============================================
            overlap = overlap_ratio(
                p_bbox,
                m_bbox
            )

            # ==============================================
            # 選最佳
            # ==============================================
            if dist < best_dist:

                best_dist = dist

                best_idx = i

                best_overlap = overlap

        # ==================================================
        # 真正 rider 條件
        # ==================================================
        if (

            best_idx is not None

            and best_dist <= adaptive_distance

            and best_overlap >= min_overlap

        ):

            used_motorcycles.add(best_idx)

            pairs.append({

                "person":
                    person,

                "motorcycle":
                    motorcycles[best_idx],

                "distance":
                    float(best_dist),

                "overlap":
                    float(best_overlap),

                "adaptive_distance":
                    float(adaptive_distance)

            })

    return pairs