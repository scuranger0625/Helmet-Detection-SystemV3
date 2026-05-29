from ultralytics import YOLO


class YOLODetector:

    def __init__(
        self,
        model_path="models/yolov8n.pt",
        conf=0.25,
        imgsz=640,
        device="cpu"
    ):

        # 載入 YOLO 模型
        self.model = YOLO(model_path)

        # 推論設定
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

        # 要保留的類別
        self.target_classes = {
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck"
        }

        # 自動轉成 class ids
        self.target_ids = [
            i for i, name in self.model.names.items()
            if name in self.target_classes
        ]

    def detect(self, frame):

        # 執行 YOLO 推論
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.target_ids,
            device=self.device,
            verbose=False
        )

        r = results[0]

        detections = []

        # 沒偵測到
        if r.boxes is None:
            return detections

        # 取出 boxes
        boxes = r.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        # 整理成統一格式
        for box, score, cls_id in zip(xyxy, confs, classes):

            detection = {
                "class_id": int(cls_id),
                "class_name": self.model.names[cls_id],
                "confidence": float(score),
                "bbox": box.tolist()
            }

            detections.append(detection)

        return detections