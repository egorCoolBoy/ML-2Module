from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from model_impl import My_LicensePlate_Model

from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def _draw_detections(frame, detections: list[dict]):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        score = det["confidence"]
        plate_text = det.get("plate_text")
        plate_text_confidence = det.get("plate_text_confidence")
        if plate_text:
            if plate_text_confidence is not None:
                label = f"{plate_text} {plate_text_confidence:.2f}"
            else:
                label = plate_text
        else:
            label = f"plate {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def run_video_mode(model: My_LicensePlate_Model, source: Path, output: Path, show: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    LOGGER.info("Video mode started | input=%s output=%s", source, output)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = model.detect_plates(frame)
        frame = _draw_detections(frame, detections)
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            LOGGER.info("Processed frames=%d", frame_idx)

        if show:
            cv2.imshow("ALPR Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    LOGGER.info("Video mode finished | frames=%d", frame_idx)


def run_camera_mode(model: My_LicensePlate_Model, camera_index: int, show: bool) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    LOGGER.info("Camera mode started | camera_index=%d", camera_index)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            LOGGER.error("Cannot read frame from camera")
            break

        detections = model.detect_plates(frame)
        frame = _draw_detections(frame, detections)
        frame_idx += 1

        if frame_idx % 30 == 0:
            LOGGER.info("Processed camera frames=%d", frame_idx)

        if show:
            cv2.imshow("ALPR Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()
    LOGGER.info("Camera mode finished | frames=%d", frame_idx)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="License plate detection CLI")
    parser.add_argument("--mode", choices=["video", "camera"], required=True)
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--source", default="data/input.mp4")
    parser.add_argument("--output", default="data/output.mp4")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR on detected plates")
    parser.add_argument(
        "--ocr-languages",
        default="en,ru",
        help="Comma-separated OCR languages for EasyOCR, for example en,ru",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        model = My_LicensePlate_Model(
            weights_path=args.weights,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            enable_ocr=not args.no_ocr,
            ocr_languages=args.ocr_languages,
        )
        if args.mode == "video":
            run_video_mode(
                model,
                source=Path(args.source),
                output=Path(args.output),
                show=args.show,
            )
        else:
            run_camera_mode(model, camera_index=args.camera_index, show=args.show)
    except Exception as exc:
        LOGGER.exception("CLI crashed: %s", exc)
        raise


if __name__ == "__main__":
    main()
