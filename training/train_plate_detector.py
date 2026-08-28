#!/usr/bin/env python
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train YOLO Plate Detector")
    parser.add_argument("--data", type=str, default="datasets/plate/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="../models/plate_detector.pt")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/plate",
        name="train",
        exist_ok=True
    )

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    import os
    final_weights = f"runs/plate/train/weights/best.pt"
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.replace(final_weights, args.save)
    print(f"Saved best model to {args.save}")


if __name__ == "__main__":
    main()
