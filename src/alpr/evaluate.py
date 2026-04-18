from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from .dataset import prepare_dataset
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate YOLO plate detector")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--dataset-source", default="My First Project.v1i.yolov8")
    parser.add_argument("--dataset-out", default="data/dataset")
    parser.add_argument("--data-yaml", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="data/eval_metrics.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    LOGGER.info("Evaluation initialization started")

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
    else:
        data_yaml = prepare_dataset(
            source_dir=args.dataset_source,
            output_dir=args.dataset_out,
        )

    model = YOLO(str(weights))
    results = model.val(data=str(data_yaml), split=args.split, device=args.device)

    metrics = {
        "mAP50": float(getattr(results.box, "map50", 0.0)),
        "mAP50-95": float(getattr(results.box, "map", 0.0)),
        "precision": float(getattr(results.box, "mp", 0.0)),
        "recall": float(getattr(results.box, "mr", 0.0)),
        "latency_ms": float(getattr(results, "speed", {}).get("inference", 0.0)),
        "split": args.split,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    LOGGER.info("Evaluation finished | metrics=%s", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
