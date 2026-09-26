from pathlib import Path
import json
import cv2

INPUT_ROOT = Path(r"C:\Projects\TinyML\dataset_detection")
OUTPUT_ROOT = Path(r"C:\Projects\TinyML\dataset_nanodet")

CLASSES = {0: "Apple", 1: "Cherry", 2: "Tomato"}


def convert_split(split, coco_name):
    image_dir = INPUT_ROOT / "images" / split
    label_dir = INPUT_ROOT / "labels" / split
    out_images = OUTPUT_ROOT / split
    out_ann = OUTPUT_ROOT / "annotations"

    out_images.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]

        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height
        })

        (out_images / image_path.name).write_bytes(image_path.read_bytes())

        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                continue

            class_id = int(values[0])
            cx, cy, bw, bh = map(float, values[1:])

            x = (cx - bw / 2) * width
            y = (cy - bh / 2) * height
            w = bw * width
            h = bh * height

            x = max(0.0, min(x, width))
            y = max(0.0, min(y, height))
            w = max(0.0, min(w, width - x))
            h = max(0.0, min(h, height - y))

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    coco = {
        "info": {"description": "Apple Cherry Tomato NanoDet dataset", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "fruit"}
            for i, name in CLASSES.items()
        ]
    }

    output = out_ann / f"instances_{coco_name}.json"
    output.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    print(f"{split}: images={len(images)}, annotations={len(annotations)}")
    print(f"JSON: {output}")


def main():
    print("Converting YOLO annotations to COCO format...")
    convert_split("train", "train")
    convert_split("validation", "val")
    print("Done.")


if __name__ == "__main__":
    main()
