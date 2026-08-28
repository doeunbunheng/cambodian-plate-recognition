#!/usr/bin/env python
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train YOLO Vehicle Detector")
    parser.add_argument("--data", type=str, default="datasets/vehicle/vehicle.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/vehicle",
        name="train",
        exist_ok=True
    )

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    model.export(format="onnx")


if __name__ == "__main__":
    main()
