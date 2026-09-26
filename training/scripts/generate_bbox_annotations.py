from pathlib import Path
import shutil

import cv2


INPUT_ROOT = Path(r"C:\Projects\TinyML\dataset")
OUTPUT_ROOT = Path(r"C:\Projects\TinyML\dataset_detection")

CLASSES = {
    "Apple": 0,
    "Cherry": 1,
    "Tomato": 2,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_fruit_bbox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(gray, 0, 245)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def bbox_to_yolo(bbox, image_width, image_height):
    x, y, w, h = bbox

    center_x = (x + w / 2.0) / image_width
    center_y = (y + h / 2.0) / image_height
    width = w / image_width
    height = h / image_height

    return center_x, center_y, width, height


def process_split(split):
    source_split = INPUT_ROOT / split
    output_images = OUTPUT_ROOT / "images" / split
    output_labels = OUTPUT_ROOT / "labels" / split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    total = 0
    successful = 0
    failed = 0

    for class_name, class_id in CLASSES.items():
        source_class = source_split / class_name

        if not source_class.exists():
            print(f"[WARNING] Missing directory: {source_class}")
            continue

        for image_path in sorted(source_class.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total += 1

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"[FAILED] Could not read: {image_path}")
                failed += 1
                continue

            image_height, image_width = image.shape[:2]

            bbox = find_fruit_bbox(image)

            if bbox is None:
                print(f"[FAILED] No bounding box: {image_path}")
                failed += 1
                continue

            center_x, center_y, width, height = bbox_to_yolo(
                bbox,
                image_width,
                image_height,
            )

            output_image = output_images / image_path.name
            output_label = output_labels / f"{image_path.stem}.txt"

            shutil.copy2(image_path, output_image)

            output_label.write_text(
                f"{class_id} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n",
                encoding="utf-8",
            )

            successful += 1

    print(
        f"{split}: total={total}, "
        f"successful={successful}, failed={failed}"
    )


def write_dataset_yaml():
    yaml_path = OUTPUT_ROOT / "dataset.yaml"

    yaml_content = """path: C:/Projects/TinyML/dataset_detection
train: images/train
val: images/validation

names:
  0: Apple
  1: Cherry
  2: Tomato
"""

    yaml_path.write_text(yaml_content, encoding="utf-8")


def main():
    print("Generating automatic bounding-box annotations...")
    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print()

    process_split("train")
    process_split("validation")

    write_dataset_yaml()

    print()
    print("Done.")
    print(f"Dataset: {OUTPUT_ROOT}")
    print(f"Config:  {OUTPUT_ROOT / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
