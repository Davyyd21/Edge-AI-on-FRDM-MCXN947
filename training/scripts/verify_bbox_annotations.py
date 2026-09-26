from pathlib import Path
import random

import cv2


DATASET_ROOT = Path(r"C:\Projects\TinyML\dataset_detection")
OUTPUT_FILE = DATASET_ROOT / "bbox_verification.jpg"

SAMPLES_PER_CLASS = 4
TILE_SIZE = 220

CLASSES = {
    0: "Apple",
    1: "Cherry",
    2: "Tomato",
}


def load_samples():
    samples = []

    for class_id, class_name in CLASSES.items():
        image_dir = DATASET_ROOT / "images" / "train"
        label_dir = DATASET_ROOT / "labels" / "train"

        class_images = []

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            line = label_path.read_text(encoding="utf-8").strip().split()

            if len(line) != 5 or int(line[0]) != class_id:
                continue

            class_images.append((image_path, label_path))

        samples.extend(
            random.sample(
                class_images,
                min(SAMPLES_PER_CLASS, len(class_images)),
            )
        )

    return samples


def draw_sample(image_path, label_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    height, width = image.shape[:2]

    values = label_path.read_text(encoding="utf-8").strip().split()

    class_id = int(values[0])
    center_x = float(values[1])
    center_y = float(values[2])
    box_width = float(values[3])
    box_height = float(values[4])

    x1 = int((center_x - box_width / 2.0) * width)
    y1 = int((center_y - box_height / 2.0) * height)
    x2 = int((center_x + box_width / 2.0) * width)
    y2 = int((center_y + box_height / 2.0) * height)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        image,
        CLASSES[class_id],
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return image


def main():
    samples = load_samples()

    if not samples:
        raise RuntimeError("No labeled samples found.")

    tiles = []

    for image_path, label_path in samples:
        image = draw_sample(image_path, label_path)

        if image is None:
            continue

        image = cv2.resize(image, (TILE_SIZE, TILE_SIZE))
        tiles.append(image)

    columns = SAMPLES_PER_CLASS
    rows = (len(tiles) + columns - 1) // columns

    sheet = 255 * __import__("numpy").ones(
        (rows * TILE_SIZE, columns * TILE_SIZE, 3),
        dtype="uint8",
    )

    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        sheet[
            row * TILE_SIZE:(row + 1) * TILE_SIZE,
            column * TILE_SIZE:(column + 1) * TILE_SIZE,
        ] = tile

    cv2.imwrite(str(OUTPUT_FILE), sheet)

    print(f"Verification image created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
