"""
Comprehensive dataset analysis + preparation for the plate recognition project.

Given the annotation structure, this builds two cleaned datasets:

1. Vehicle Dataset (detection)  - uses LARGE boxes (vehicle-level)
2. Plate Dataset (detection)    - uses SMALL boxes (plate + text boxes)

Classes with < 5 examples are dropped from training (can't train YOLO on them).
"""
import os
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict

import yaml

IMAGES_DIR = Path(r"D:\Training_model\Dataset\images")
LABELS_DIR = Path(r"D:\Training_model\Dataset\labels")
DATA_YAML = Path(r"D:\Training_model\Dataset\data.yaml")

OUTPUT_DIR = Path(r"D:\Training_model\cambodian-plate-recognition\training\datasets")

# Thresholds (normalized box size)
VEHICLE_THRESHOLD = 0.3   # boxes with w or h >= this are vehicle-level
PLATE_MAX_W = 0.2         # boxes with w <= this are plate/text level

# Splits
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


def load_class_names():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"]


def parse_label_file(path: Path):
    annotations = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                annotations.append({
                    "class_id": int(parts[0]),
                    "cx": float(parts[1]),
                    "cy": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                })
    return annotations


def main():
    random.seed(42)
    class_names = load_class_names()

    images = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Total images: {len(images)}")

    vehicle_samples = []   # (image_path, box, class_name)
    plate_samples = []     # (image_path, box, class_name or 'plate')

    # Pass 1: categorize each label per image
    for img_path in images:
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        annotations = parse_label_file(label_path)
        if not annotations:
            continue

        # A box is vehicle-level if ANY box is large
        has_vehicle = any(
            a["w"] >= VEHICLE_THRESHOLD or a["h"] >= VEHICLE_THRESHOLD
            for a in annotations
        )

        if has_vehicle:
            # Pick the largest box as vehicle
            vehicle_box = max(annotations, key=lambda a: a["w"] * a["h"])
            cls_name = class_names.get(vehicle_box["class_id"], "vehicle")
            vehicle_samples.append({
                "img": str(img_path),
                "box": vehicle_box,
                "class": cls_name,
            })

        # Plate-level: small boxes that are NOT the vehicle box
        for a in annotations:
            if a["w"] >= VEHICLE_THRESHOLD or a["h"] >= VEHICLE_THRESHOLD:
                continue
            if has_vehicle and a == vehicle_box:
                continue
            cls_name = class_names.get(a["class_id"], "text")
            plate_samples.append({
                "img": str(img_path),
                "box": a,
                "class": cls_name,
            })

    print(f"\nVehicle-level samples: {len(vehicle_samples)}")
    print(f"Plate/text-level samples: {len(plate_samples)}")

    # Analyze vehicle classes
    print("\n=== VEHICLE CLASSES (by count) ===")
    vc = defaultdict(int)
    for s in vehicle_samples:
        vc[s["class"]] += 1
    for name, count in sorted(vc.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    # Analyze plate classes - which are provinces vs numbers
    print("\n=== PLATE/PC CLASSES (provinces) ===")
    pc = defaultdict(int)
    pn = defaultdict(int)
    for s in plate_samples:
        name = s["class"].lower()
        if "pc" in name or any(p.lower() in name for p in
                               ["phnompenh", "siemreap", "kandal", "battambang",
                                "kampot", "kep", "pursat", "kratie", "takeo",
                                "sihanouk", "kampong", "prey", "svay", "ratanakiri",
                                "mundul", "tboung", "banteay", "oddar", "preah",
                                "stung", "koh", "pailin"]):
            pc[s["class"]] += 1
        else:
            pn[s["class"]] += 1

    print(f"  Total PC (province) annotations: {sum(pc.values())}")
    for name, count in sorted(pc.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    print(f"\n  Total PN (number) annotations: {sum(pn.values())}")
    print(f"  Unique PN classes: {len(pn)}")
    print(f"  (PN classes will be handled by OCR, not YOLO)")

    # Build usable datasets
    build_datasets(vehicle_samples, plate_samples, class_names)


def build_datasets(vehicle_samples, plate_samples, class_names):
    """Build YOLO-format datasets with train/val split."""

    # ===== VEHICLE DATASET =====
    # Only keep classes with >= 5 examples
    vehicle_counts = defaultdict(int)
    for s in vehicle_samples:
        vehicle_counts[s["class"]] += 1
    usable_vehicle_classes = {c for c, n in vehicle_counts.items() if n >= 5}

    print(f"\n=== BUILDING VEHICLE DATASET ===")
    print(f"Classes with >=5 examples: {sorted(usable_vehicle_classes)}")

    vehicle_class_map = {c: i for i, c in enumerate(sorted(usable_vehicle_classes))}

    veh_out = OUTPUT_DIR / "vehicle"
    for split in ("train", "val"):
        (veh_out / "images" / split).mkdir(parents=True, exist_ok=True)
        (veh_out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Group vehicle samples by image to avoid duplicates
    img_groups = defaultdict(list)
    for s in vehicle_samples:
        if s["class"] not in usable_vehicle_classes:
            continue
        img_groups[s["img"]].append(s)

    images = list(img_groups.keys())
    random.shuffle(images)
    n_train = int(len(images) * TRAIN_RATIO)

    v_train_count = 0
    v_val_count = 0
    for i, img in enumerate(images):
        split = "train" if i < n_train else "val"
        img_name = Path(img).name
        shutil.copy(img, veh_out / "images" / split / img_name)

        # Write label
        label_path = veh_out / "labels" / split / (Path(img).stem + ".txt")
        lines = []
        for s in img_groups[img]:
            b = s["box"]
            cls_id = vehicle_class_map[s["class"]]
            lines.append(f"{cls_id} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        if split == "train":
            v_train_count += 1
        else:
            v_val_count += 1

    # Write vehicle data.yaml
    veh_yaml = veh_out / "data.yaml"
    veh_yaml.write_text(
        f"path: {veh_out}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        + "".join(f"  {i}: {name}\n" if " " not in name else f"  {i}: '{name}'\n"
                  for name, i in sorted(vehicle_class_map.items(), key=lambda x: x[1]))
    )
    print(f"Vehicle dataset: {v_train_count} train, {v_val_count} val, {len(usable_vehicle_classes)} classes")

    # ===== PLATE DATASET =====
    # Focus on PLATE detection: all small boxes become class 'plate'
    # (do NOT classify by number - that's OCR's job)
    print(f"\n=== BUILDING PLATE DATASET ===")
    plate_count = 0
    plate_out = OUTPUT_DIR / "plate"
    for split in ("train", "val"):
        (plate_out / "images" / split).mkdir(parents=True, exist_ok=True)
        (plate_out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Group by image
    plate_imgs = defaultdict(list)
    for s in plate_samples:
        plate_imgs[s["img"]].append(s)

    all_plate_imgs = list(plate_imgs.keys())
    random.shuffle(all_plate_imgs)
    n_train_plate = int(len(all_plate_imgs) * TRAIN_RATIO)

    for i, img in enumerate(all_plate_imgs):
        split = "train" if i < n_train_plate else "val"
        img_name = Path(img).name
        shutil.copy(img, plate_out / "images" / split / img_name)

        # Every box becomes class 0 'plate'
        label_path = plate_out / "labels" / split / (Path(img).stem + ".txt")
        lines = []
        for s in plate_imgs[img]:
            b = s["box"]
            lines.append(f"0 {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        plate_count += 1

    plate_yaml = plate_out / "data.yaml"
    plate_yaml.write_text(
        f"path: {plate_out}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        f"  0: plate\n"
    )
    print(f"Plate dataset: {plate_count} images, 1 class (plate)\n")

    print("=== DATASETS READY ===")
    print(f"Vehicle: {veh_out}")
    print(f"Plate: {plate_out}")


if __name__ == "__main__":
    main()
