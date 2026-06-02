import time
import cv2
import os
import numpy as np

from core.scheduler import SystemState
from core.tracker import pair_person_motorcycle
from core.head_crop import crop_head

from utils.drawing import draw_detections


def compute_budget_pressure(
    yolo_runs,
    history_size=30
):

    if len(yolo_runs) == 0:
        return 0.0

    recent = yolo_runs[-history_size:]

    pressure = sum(recent) / len(recent)

    return float(np.clip(pressure, 0.0, 1.0))


def process_frame(
    frame,
    frame_idx,
    runtime,
    modules
):

    frame_start = time.time()

    fps = modules.fps_counter.update()

    preprocess_start = time.time()

    motion_score = (
        modules.motion_estimator.compute_motion(frame)
    )

    preprocess_ms = (
        time.time() - preprocess_start
    ) * 1000.0

    classifier_entropy = 0.0

    tracker_confidence = 1.0

    budget_pressure = compute_budget_pressure(
        runtime.yolo_history
    )

    state = SystemState(
        motion_score=motion_score,
        tracker_confidence=tracker_confidence,
        classifier_entropy=classifier_entropy,
        budget_pressure=budget_pressure
    )

    should_detect, urgency, reason = (
        modules.scheduler.should_detect(state)
    )

    infer_ms = 0.0

    if should_detect:

        infer_start = time.time()

        detections = (
            modules.detector.detect(frame)
        )

        infer_ms = (
            time.time() - infer_start
        ) * 1000.0

        runtime.cached_detections = detections

        runtime.yolo_history.append(1)

        runtime.yolo_calls += 1

    else:

        detections = runtime.cached_detections

        runtime.yolo_history.append(0)

    pairs = pair_person_motorcycle(
        detections
    )
    frame_violations = 0

    runtime.person_count = sum(
        1 for det in detections
        if det.get("class_name") == "person"
    )

    runtime.motorcycle_count = sum(
        1 for det in detections
        if det.get("class_name") == "motorcycle"
    )

    runtime.rider_pair_count = len(pairs)

    runtime.rider_count = len(pairs)

    for pair_idx, pair in enumerate(pairs):

        person = pair["person"]

        px1, py1, px2, py2 = map(
            int,
            person["bbox"]
        )

        person_crop = frame[
            py1:py2,
            px1:px2
        ]

        if person_crop.size == 0:
            continue

        head_detections = (
            modules.head_detector.detect(person_crop)
        )

        if len(head_detections) == 0:

            result = crop_head(
                frame,
                person["bbox"]
            )

            if result is None:
                continue

            head_crop, head_bbox = result

        else:

            det = head_detections[0]

            hx1, hy1, hx2, hy2 = map(
                int,
                det["bbox"]
            )

            gx1 = px1 + hx1
            gy1 = py1 + hy1

            gx2 = px1 + hx2
            gy2 = py1 + hy2

            head_bbox = (
                gx1,
                gy1,
                gx2,
                gy2
            )

            head_crop = frame[
                gy1:gy2,
                gx1:gx2
            ]

            if head_crop.size == 0:
                continue

        hx1, hy1, hx2, hy2 = head_bbox

        os.makedirs(
            "dataset/raw_heads",
            exist_ok=True
        )

        if frame_idx % 5 == 0:

            filename = (
                f"dataset/raw_heads/"
                f"head_{frame_idx}_{pair_idx}.jpg"
            )

            cv2.imwrite(
                filename,
                head_crop
            )

        cls_result = (
            modules.classifier.estimate(head_crop)
        )

        status = cls_result["status"]

        score = cls_result["risk_score"]

        helmet_score = cls_result["helmet_score"]

        if status == "NO HELMET":

            color = (0, 0, 255)

            label = f"NO HELMET {score:.2f}"

            frame_violations += 1

        elif status == "SUSPICIOUS":

            color = (0, 165, 255)

            label = f"SUSPICIOUS {score:.2f}"

        else:

            color = (255, 0, 0)

            label = f"HELMET {helmet_score:.2f}"

        cv2.rectangle(
            frame,
            (hx1, hy1),
            (hx2, hy2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (hx1, max(30, hy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    frame = draw_detections(
        frame,
        detections
    )

    draw_start = time.time()

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"YOLO RUN: {should_detect}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"MOTION: {motion_score:.4f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"URGENCY: {urgency:.3f}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 128, 0),
        2
    )

    cv2.putText(
        frame,
        f"VIOLATIONS: {runtime.violation_frame_count}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    draw_ms = (
        time.time() - draw_start
    ) * 1000.0

    total_ms = (
        time.time() - frame_start
    ) * 1000.0

    runtime.preprocess_ms = preprocess_ms
    runtime.infer_ms = infer_ms
    runtime.draw_ms = draw_ms
    runtime.total_ms = total_ms

    runtime.motion_score = motion_score
    runtime.urgency_score = urgency
    runtime.last_yolo_run = should_detect


    # =====================================================
    # Violation Cooldown System
    # =====================================================

    if not hasattr(runtime, "last_violation_frame"):
        runtime.last_violation_frame = -999

    VIOLATION_COOLDOWN = 30

    if frame_violations > 0:

        if (
            frame_idx - runtime.last_violation_frame
            > VIOLATION_COOLDOWN
        ):

            runtime.violation_count += 1

            runtime.last_violation_frame = frame_idx

    runtime.violation_frame_count += frame_violations

    modules.evaluator.update(
        fps=fps,
        yolo_run=should_detect,
        frame_violations=frame_violations,
        preprocess_ms=preprocess_ms,
        infer_ms=infer_ms,
        draw_ms=draw_ms,
        total_ms=total_ms
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"YOLO RUN: {should_detect}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"MOTION: {motion_score:.4f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"URGENCY: {urgency:.3f}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 128, 0),
        2
    )

    cv2.putText(
        frame,
        f"VIOLATIONS: {runtime.violation_count}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    return frame
