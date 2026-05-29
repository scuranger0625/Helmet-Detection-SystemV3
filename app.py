import cv2
import uvicorn
import os
import asyncio

from pathlib import Path

from fastapi import FastAPI
from fastapi import Request
from fastapi import File
from fastapi import UploadFile

from fastapi.responses import (
    StreamingResponse,
    HTMLResponse,
    JSONResponse
)

from fastapi.templating import (
    Jinja2Templates
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from runtime.modules import (
    InferenceModules
)

from runtime.runtime import (
    RuntimeState
)

from runtime.pipeline import (
    process_frame
)

# =====================================================
# Base Directory
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# FastAPI
# =====================================================

app = FastAPI(
    title="Helmet Detection System"
)

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Templates
# =====================================================

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

# =====================================================
# Video Paths
# =====================================================

VIDEO_DIR = str(
    BASE_DIR / "video"
)

VIDEO_SOURCE = str(
    BASE_DIR
    / "video"
    / "input_video.mp4"
)

os.makedirs(
    VIDEO_DIR,
    exist_ok=True
)

# =====================================================
# Global Stats
# =====================================================

global_stats = {

    "fps": 0.0,

    "violations": 0,

    "motion": 0.0,

    "urgency": 0.0,

    "fps": 0.0,
    "fps_avg": 0.0,
    "fps_min": 0.0,
    "fps_p5": 0.0,
    "fps_p95": 0.0,
    "violations": 0,
    "frame_violations": 0,
    "motion": 0.0,
    "urgency": 0.0,
    "yolo_run": False,
    "person_count": 0,
    "motorcycle_count": 0,
    "rider_pair_count": 0,
    "rider_count": 0,
    "yolo_calls": 0,
    "total_frames": 0,
    "saved_ratio": 0.0,
    "detect_ratio": 0.0,
    "cache_hit_rate": 0.0,
    "detector_calls_per_second": 0.0,
    "detector_calls_per_minute": 0.0,
    "budget_pressure": 0.0,
    "avg_inference_ms": 0.0,
    "preprocess_ms_avg": 0.0,
    "infer_ms_avg": 0.0,
    "draw_ms_avg": 0.0,
    "total_ms_avg": 0.0,
    "event_violations": 0
}

# =====================================================
# Runtime Modules
# =====================================================

modules = InferenceModules()

runtime = RuntimeState()

# =====================================================
# Home Page
# =====================================================

@app.get(
    "/",
    response_class=HTMLResponse
)

async def index(
    request: Request
):

    return templates.TemplateResponse(

        request=request,

        name="index.html",

        context={
            "request": request
        }
    )

# =====================================================
# Upload Video
# =====================================================

@app.post("/upload_video")

async def upload_video(

    file: UploadFile = File(...)
):

    try:

        content = await file.read()

        with open(
            VIDEO_SOURCE,
            "wb"
        ) as f:

            f.write(content)

            f.flush()

            os.fsync(f.fileno())

        # reset runtime
        modules.reset(runtime)

        global_stats["fps"] = 0.0

        global_stats["violations"] = 0

        print("✅ 新影片已載入")

        return {

            "status": "success"
        }

    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={
                "message": str(e)
            }
        )

# =====================================================
# Stats
# =====================================================

@app.get("/stats")
async def stats():

    return await get_stats()

# =====================================================
# Pause / Resume
# =====================================================

@app.post("/pause")

async def pause_video():

    global paused

    paused = True

    return {
        "status": "paused"
    }


@app.post("/resume")

async def resume_video():

    global paused

    paused = False

    return {
        "status": "running"
    }


async def get_stats():

    return global_stats

# =====================================================
# Frame Generator
# =====================================================

async def gen_frames():

    global global_stats

    await asyncio.sleep(1.0)

    if not os.path.exists(
        VIDEO_SOURCE
    ):
        return

    cap = cv2.VideoCapture(
        VIDEO_SOURCE
    )

    frame_idx = 0

    encode_param = [

        int(cv2.IMWRITE_JPEG_QUALITY),

        80
    ]

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_idx += 1

        try:

            output_frame = process_frame(

                frame,

                frame_idx,

                runtime,

                modules
            )

        except Exception as e:

            print(
                f"辨識出錯！"
                f"{frame.shape[1]}x"
                f"{frame.shape[0]}"
            )

            print(
                f"錯誤原因: {e}"
            )

            output_frame = frame

        # update stats
        global_stats["fps"] = round(

            modules.fps_counter.fps

            if hasattr(
                modules.fps_counter,
                "fps"
            )

            else 0.0,

            1
        )

        global_stats["violations"] = (
            runtime.violation_count
        )

        # ==================================================
        # Motion Score
        # ==================================================
        global_stats["motion"] = round(

            runtime.motion_score,

            3

        ) if hasattr(
            runtime,
            "motion_score"
        ) else 0.0

        # ==================================================
        # Urgency Score
        # ==================================================
        global_stats["urgency"] = round(

            runtime.urgency_score,

            3

        ) if hasattr(
            runtime,
            "urgency_score"
        ) else 0.0

        # ==================================================
        # YOLO 是否執行
        # ==================================================
        global_stats["yolo_run"] = (

            runtime.last_yolo_run

        ) if hasattr(
            runtime,
            "last_yolo_run"
        ) else False

        # ==================================================
        # Rider Count
        # ==================================================
        global_stats["rider_count"] = (

            runtime.rider_count

        ) if hasattr(
            runtime,
            "rider_count"
        ) else 0

        # ==================================================
        # YOLO 呼叫次數
        # ==================================================
        global_stats["yolo_calls"] = (

            runtime.yolo_calls

        ) if hasattr(
            runtime,
            "yolo_calls"
        ) else 0

        # ==================================================
        # Total Frames
        # ==================================================
        global_stats["total_frames"] = frame_idx

        # ==================================================
        # Saved Compute Ratio
        # ==================================================
        if frame_idx > 0:

            saved_ratio = 1.0 - (

                global_stats["yolo_calls"]

                / frame_idx
            )

            global_stats["saved_ratio"] = round(

                saved_ratio * 100,

                1
            )

        # ==================================================
        # Scheduler Metrics
        # ==================================================
        scheduler_metrics = modules.scheduler.get_metrics()

        global_stats["detect_ratio"] = round(
            scheduler_metrics.get("detect_ratio", 0.0) * 100,
            1
        )

        global_stats["cache_hit_rate"] = round(
            scheduler_metrics.get("reuse_ratio", 0.0) * 100,
            1
        )

        global_stats["detector_calls_per_second"] = round(
            scheduler_metrics.get("detector_calls_per_second", 0.0),
            2
        )

        global_stats["detector_calls_per_minute"] = round(
            scheduler_metrics.get("detector_calls_per_minute", 0.1),
            1
        )

        global_stats["budget_pressure"] = round(
            scheduler_metrics.get("last_budget_pressure", 0.0),
            2
        )

        # ==================================================
        # Inference Latency Metrics
        # ==================================================
        evaluator_summary = modules.evaluator.summary()

        global_stats["fps_avg"] = round(
            evaluator_summary.get("avg_fps", 0.0),
            1
        )
        global_stats["fps_min"] = round(
            evaluator_summary.get("min_fps", 0.0),
            1
        )
        global_stats["fps_p5"] = round(
            evaluator_summary.get("p5_fps", 0.0),
            1
        )
        global_stats["fps_p95"] = round(
            evaluator_summary.get("p95_fps", 0.0),
            1
        )

        global_stats["preprocess_ms_avg"] = round(
            evaluator_summary.get("preprocess_ms_avg", 0.0),
            1
        )
        global_stats["infer_ms_avg"] = round(
            evaluator_summary.get("infer_ms_avg", 0.0),
            1
        )
        global_stats["draw_ms_avg"] = round(
            evaluator_summary.get("draw_ms_avg", 0.0),
            1
        )
        global_stats["total_ms_avg"] = round(
            evaluator_summary.get("total_ms_avg", 0.0),
            1
        )

        global_stats["event_violations"] = (
            runtime.violation_count
        ) if hasattr(
            runtime,
            "violation_count"
        ) else 0

        global_stats["frame_violations"] = (
            runtime.violation_frame_count
        ) if hasattr(
            runtime,
            "violation_frame_count"
        ) else 0

        global_stats["person_count"] = (
            runtime.person_count
        ) if hasattr(
            runtime,
            "person_count"
        ) else 0

        global_stats["motorcycle_count"] = (
            runtime.motorcycle_count
        ) if hasattr(
            runtime,
            "motorcycle_count"
        ) else 0

        global_stats["rider_pair_count"] = (
            runtime.rider_pair_count
        ) if hasattr(
            runtime,
            "rider_pair_count"
        ) else 0

        # ==================================================
        # Avg Inference Time
        # ==================================================
        global_stats["avg_inference_ms"] = round(

            runtime.avg_inference_ms,

            1

        ) if hasattr(
            runtime,
            "avg_inference_ms"
        ) else 0.0

        # encode jpg
        ret, buffer = cv2.imencode(

            ".jpg",

            output_frame,

            encode_param
        )

        if not ret:
            continue

        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + buffer.tobytes()

            + b"\r\n"
        )

        await asyncio.sleep(0.005)

    cap.release()

# =====================================================
# Video Feed
# =====================================================

@app.get("/video_feed")

async def video_feed():

    return StreamingResponse(

        gen_frames(),

        media_type=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )

# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8000
    )