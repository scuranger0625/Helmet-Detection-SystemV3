import cv2


def draw_detections(frame, detections):

    for det in detections:

        x1, y1, x2, y2 = map(int, det["bbox"])

        class_name = det["class_name"]
        conf = det["confidence"]

        label = f"{class_name} {conf:.2f}"

        # 畫框
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 顯示文字
        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return frame