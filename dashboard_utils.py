from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import yt_dlp


DEFAULT_MODEL_PATH = "model/bestone.pt"


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def list_media_files(folder: Path, extensions: set[str]) -> list[str]:
    if not folder.exists():
        return []
    return sorted(str(path) for path in folder.iterdir() if path.suffix.lower() in extensions)


def save_uploaded_file(uploaded_file, suffix: str) -> str:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(uploaded_file.getbuffer())
    handle.flush()
    handle.close()
    return handle.name


def _class_counts(result) -> dict[str, int]:
    counts = Counter()
    if result.boxes is None or len(result.boxes) == 0:
        return {}

    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    for class_id in class_ids:
        class_name = result.names.get(int(class_id), str(int(class_id)))
        counts[class_name] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def render_frame(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def get_youtube_stream(youtube_url: str) -> str | None:
    """Return a direct playable stream URL for a YouTube link using yt_dlp."""
    ydl_opts = {
        'format': 'bestvideo[height>=480][ext=mp4]/bestvideo[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info.get('url')


def analyze_image(model: YOLO, image_bytes: bytes, conf: float, imgsz: int, iou: float = 0.45) -> dict[str, object]:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(image_bytes)
    temp_file.flush()
    temp_file.close()

    try:
        pil_image = Image.open(temp_file.name).convert("RGB")
        image_array = np.array(pil_image)
        results = model.predict(source=image_array, verbose=False, conf=conf, imgsz=imgsz, iou=iou)
        result = results[0]
        annotated = render_frame(result.plot())
        class_counts = _class_counts(result)

        return {
            "input_image": image_array,
            "annotated_image": annotated,
            "detections": len(result.boxes),
            "class_counts": class_counts,
        }
    finally:
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass


def analyze_video(
    model: YOLO,
    source_path: str,
    conf: float,
    imgsz: int,
    max_frames: int,
    frame_stride: int,
    iou: float = 0.45,
) -> dict[str, object]:
    capture = cv2.VideoCapture(source_path)
    preview_frames: list[np.ndarray] = []
    class_counts = Counter()
    total_detections = 0
    frames_processed = 0
    frame_index = 0

    while capture.isOpened() and frames_processed < max_frames:
        success, frame = capture.read()
        if not success:
            break

        frame_index += 1
        if frame_index % frame_stride != 0:
            continue

        results = model.predict(source=frame, verbose=False, conf=conf, imgsz=imgsz, iou=iou)
        result = results[0]
        total_detections += len(result.boxes)
        class_counts.update(_class_counts(result))
        annotated = render_frame(result.plot())

        if len(preview_frames) < 4:
            preview_frames.append(annotated)

        frames_processed += 1

    capture.release()

    avg_detections = total_detections / frames_processed if frames_processed else 0.0

    return {
        "preview_frames": preview_frames,
        "stats": {
            "frames_processed": frames_processed,
            "total_detections": total_detections,
            "avg_detections": avg_detections,
            "class_counts": dict(sorted(class_counts.items(), key=lambda item: item[0])),
        },
    }