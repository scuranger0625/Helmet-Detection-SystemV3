from pathlib import Path
import cv2


def open_video(source):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return capture


def list_mp4_files(folder: str | Path):
    return sorted(Path(folder).glob("*.mp4"))
