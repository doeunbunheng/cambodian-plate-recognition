import os
import shutil
from pathlib import Path

SOURCE_IMAGES = Path(r"D:\Training_model\Dataset\images")
SOURCE_LABELS = Path(r"D:\Training_model\Dataset\labels")
OUTPUT_ROOT = Path(r"D:\Training_model\cambodian-plate-recognition\training\datasets")


def prepare_datasets():
    output_root = OUTPUT_ROOT

    vehicle_root = output_root / "vehicle"
    plate_root = output_root / "plate"

    vehicle_images = vehicle_root / "images"
    vehicle_labels = vehicle_root / "labels"
    plate_images = plate_root / "images"
    plate_labels = plate_root / "labels"

    for d in [vehicle_images / "train", vehicle_images / "val",
              vehicle_labels / "train", vehicle_labels / "val",
              plate_images / "train", plate_images / "val",
              plate_labels / "train", plate_labels / "val"]:
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(SOURCE_IMAGES.glob("*.jpg"))
    print(f"Total images: {len(images)}")

    train_count = int(len(images) * 0.8)
    val_count = int(len(images) * 0.2)

    for idx, img_path in enumerate(images):
        label_path = SOURCE_LABELS / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        split = "train" if idx < train_count else "val"

        shutil.copy(img_path, vehicle_images / split / img_path.name)
        shutil.copy(img_path, plate_images / split / img_path.name)

        if label_path.exists():
            shutil.copy(label_path, vehicle_labels / split / label_path.name)
            shutil.copy(label_path, plate_labels / split / label_path.name)

    print(f"Prepared {train_count} train, {len(images) - train_count} val images")

    write_data_yamls()


def write_data_yamls():
    vehicle_yaml = OUTPUT_ROOT / "vehicle" / "vehicle.yaml"
    plate_yaml = OUTPUT_ROOT / "plate" / "plate.yaml"

    vehicle_yaml.write_text(f"""# Vehicle Detection Dataset
path: {OUTPUT_ROOT / 'vehicle'}
train: images/train
val: images/val

names:
  0: FamilyCar
  1: Taxi
  2: Bus
  3: Truck
  4: Motorcycle
  5: TukTuk
  6: Van
""")

    plate_yaml.write_text(f"""# Plate Detection Dataset
path: {OUTPUT_ROOT / 'plate'}
train: images/train
val: images/val

# 4 keypoints: top-left, top-right, bottom-right, bottom-left
kpt_shape: [4, 2]

names:
  0: plate
""")

    print("Written data.yaml files")


if __name__ == "__main__":
    prepare_datasets()
