from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO
import wandb

from .dataset import prepare_dataset
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO for plate detection")
    parser.add_argument("--dataset-source", default="My First Project.v1i.yolov8")
    parser.add_argument("--dataset-out", default="data/dataset")
    parser.add_argument("--data-yaml", default="")
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="plate_detector")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ml2-license-plate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    LOGGER.info("Training initialization started")

    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
    else:
        data_yaml = prepare_dataset(
            source_dir=args.dataset_source,
            output_dir=args.dataset_out,
        )

    model = YOLO(args.base_model)
    train_results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    run_dir = Path(getattr(train_results, "save_dir", Path(args.project) / args.name))
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"best.pt was not produced: {best_weights}")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    final_weights = model_dir / "best.pt"
    shutil.copy2(best_weights, final_weights)

    val_results = model.val(data=str(data_yaml), split="test", device=args.device)
    metrics = {
        "mAP50": float(getattr(val_results.box, "map50", 0.0)),
        "mAP50-95": float(getattr(val_results.box, "map", 0.0)),
        "latency_ms": float(getattr(val_results, "speed", {}).get("inference", 0.0)),
    }

    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.name,
            config={
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "base_model": args.base_model,
            },
        )
        run.log(metrics)
        artifact = wandb.Artifact("best-weights", type="model")
        artifact.add_file(str(final_weights))
        run.log_artifact(artifact)
        run.finish()

    LOGGER.info("Training finished | best_weights=%s", final_weights)
    LOGGER.info("Validation metrics | %s", metrics)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
