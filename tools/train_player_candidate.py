"""Reproducible small real-frame player detector experiment; never auto-deploy."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import cv2
import torch
from ultralytics import YOLO
from starfall.training import select_device, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        help="Reviewed player dataset YAML; auto-discovers the latest saved split",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    candidates = sorted(Path("data/player-reviewed").glob("split-*/dataset.yaml"))
    if not args.dataset and not candidates:
        parser.error(
            "Import the training-data package or supply --dataset with a reviewed split"
        )
    source = Path(args.dataset) if args.dataset else candidates[-1]
    device = select_device(args.device)
    name = "player-reviewed-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    data = str(
        prepare_dataset(
            source, Path("models/detector-experiments") / name / "dataset.yaml"
        )
    )
    model = YOLO("yolo26n.pt")
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=320,
        batch=args.batch,
        nbs=args.batch,
        device=device,
        workers=0,
        freeze=10,
        seed=42,
        deterministic=True,
        project=str(Path("models/detector-experiments").resolve()),
        name=name,
        exist_ok=True,
        plots=False,
        optimizer="AdamW",
        lr0=0.003,
        warmup_epochs=0,
        mosaic=0,
        mixup=0,
        fliplr=0,
        scale=0.2,
        hsv_s=0.2,
        hsv_v=0.2,
        patience=15,
        verbose=False,
    )
    checkpoint = model.trainer.save_dir / "weights" / "best.pt"
    model = YOLO(checkpoint)
    metrics = model.val(
        data=data,
        split="test",
        imgsz=320,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
    )
    report = {
        "scope": "small_player_only_reviewed_proposals",
        "device": device,
        "source_dataset": str(source),
        "independent_test": "separate attempt; cross-attempt near duplicates checked",
        "limitations": "31 positives visually reviewed by Codex; template-based selection bias; approximate boxes; no hazards; not exhaustive game accuracy",
        "metrics": metrics.results_dict,
        "speed": metrics.speed,
        "dataset": data,
        "checkpoint": str(checkpoint),
        "deployed": False,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/player-detector-candidate.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
