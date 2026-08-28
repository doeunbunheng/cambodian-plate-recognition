import os
from pathlib import Path
from collections import defaultdict

LABELS_DIR = Path(r"D:\Training_model\Dataset\labels")

class_stats = defaultdict(lambda: {"count": 0, "boxes": [], "files": set()})


def main():
    label_files = sorted(LABELS_DIR.glob("*.txt"))

    for label_file in label_files:
        if label_file.name in ("classes.txt",):
            continue

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                class_stats[class_id]["count"] += 1
                class_stats[class_id]["boxes"].append((w, h))
                class_stats[class_id]["files"].add(label_file.name)

    print(f"{'Class':<5} {'Count':<8} {'Avg W':<8} {'Avg H':<8} {'Platform':<15} {'Diagnosis'}")
    print("=" * 70)

    for class_id in sorted(class_stats.keys()):
        stats = class_stats[class_id]
        count = stats["count"]
        if count == 0:
            continue

        boxes = stats["boxes"]
        avg_w = sum(b[0] for b in boxes) / len(boxes)
        avg_h = sum(b[1] for b in boxes) / len(boxes)
        n_files = len(stats["files"])

        if avg_w > 0.3 or avg_h > 0.3:
            platform = "VEHICLE/LARGE"
        elif avg_w < 0.15 and avg_h < 0.05:
            platform = "PLATE-TEXT"
        elif avg_w < 0.2 and avg_h < 0.1:
            platform = "PLATE-BOX"
        else:
            platform = "PLATE-LARGE?"

        print(f"{class_id:<5} {count:<8} {avg_w:<8.4f} {avg_h:<8.4f} {platform:<15} {n_files} files")


if __name__ == "__main__":
    main()
