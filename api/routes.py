import cv2
import io
import base64
import numpy as np

from PIL import Image

from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File
from fastapi.responses import JSONResponse

from ultralytics import YOLO

router = APIRouter()

model = YOLO("../models/yolov8n.pt")

@router.get("/")
def root():

    return {
        "message": "YOLOv8 FastAPI backend is running!"
    }

@router.post("/detect")
async def detect_object(
    file: UploadFile = File(...)
):

    try:

        image_bytes = await file.read()

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image = np.array(pil_image)

        results = model.predict(image)

        detections = []

        for r in results:

            for box in r.boxes:

                cls = int(box.cls)

                label = model.names[cls]

                conf = float(box.conf)

                bbox = [
                    float(x)
                    for x in box.xyxy[0]
                ]

                detections.append({

                    "class": label,

                    "confidence": conf,

                    "bbox": bbox
                })

                x1, y1, x2, y2 = map(
                    int,
                    bbox
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

        _, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )
        )

        img_base64 = base64.b64encode(
            buffer
        ).decode("utf-8")

        return JSONResponse(content={

            "detections": detections,

            "annotated_image": img_base64
        })

    except Exception as e:

        return JSONResponse(
            content={
                "error": str(e)
            },
            status_code=400
        )