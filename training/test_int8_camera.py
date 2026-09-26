import os
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


MODEL = r"C:\Projects\TinyML\nanodet\nanodet_m_0.5x_fruits_96_tflite\nanodet_m_0.5x_fruits_96_int8_camera.tflite"

DATASET_DIR = r"C:\Projects\TinyML\dataset\train"

IMAGE_SIZE = 96

CLASS_NAMES = [
    "Apple",
    "Cherry",
    "Tomato"
]

STRIDES = [8, 16, 32]

REG_MAX = 7

NUM_CLASSES = 3

SCORE_THRESHOLD = 0.05

NMS_THRESHOLD = 0.60


def preprocess(image_path, input_scale, input_zero_point):
    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (320, 240),
        interpolation=cv2.INTER_LINEAR
    )

    r5 = (image[:, :, 0] >> 3).astype(np.uint8)
    g6 = (image[:, :, 1] >> 2).astype(np.uint8)
    b5 = (image[:, :, 2] >> 3).astype(np.uint8)

    rgb565 = (
        (r5.astype(np.uint16) << 11)
        | (g6.astype(np.uint16) << 5)
        | b5.astype(np.uint16)
    )

    r = ((rgb565 >> 11) & 0x1F).astype(np.uint8)
    g = ((rgb565 >> 5) & 0x3F).astype(np.uint8)
    b = (rgb565 & 0x1F).astype(np.uint8)

    r = (
        r.astype(np.uint16) * 255 // 31
    ).astype(np.uint8)

    g = (
        g.astype(np.uint16) * 255 // 63
    ).astype(np.uint8)

    b = (
        b.astype(np.uint16) * 255 // 31
    ).astype(np.uint8)

    restored = np.stack(
        [r, g, b],
        axis=-1
    )

    resized = cv2.resize(
        restored,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    mean = np.array(
        [103.53, 116.28, 123.675],
        dtype=np.float32
    )

    std = np.array(
        [57.375, 57.12, 58.395],
        dtype=np.float32
    )

    normalized = (
        resized.astype(np.float32) - mean
    ) / std

    quantized = np.round(
        normalized / input_scale
        + input_zero_point
    )

    quantized = np.clip(
        quantized,
        -128,
        127
    ).astype(np.int8)

    return quantized


def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def distribution_integral(values):
    values = np.asarray(values, dtype=np.float32)

    probabilities = sigmoid(values)

    probabilities = probabilities.reshape(4, REG_MAX + 1)

    result = np.zeros(4, dtype=np.float32)

    bins = np.arange(
        REG_MAX + 1,
        dtype=np.float32
    )

    for i in range(4):
        p = probabilities[i]

        total = np.sum(p)

        if total > 0.0:
            p = p / total

        result[i] = np.sum(
            p * bins
        )

    return result


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1
    ) * max(
        0.0,
        ay2 - ay1
    )

    area_b = max(
        0.0,
        bx2 - bx1
    ) * max(
        0.0,
        by2 - by1
    )

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def apply_nms(detections):
    detections = sorted(
        detections,
        key=lambda x: x["score"],
        reverse=True
    )

    selected = []

    while detections:
        current = detections.pop(0)

        selected.append(current)

        remaining = []

        for candidate in detections:
            if candidate["class_id"] != current["class_id"]:
                remaining.append(candidate)
                continue

            overlap = iou(
                current["box"],
                candidate["box"]
            )

            if overlap < NMS_THRESHOLD:
                remaining.append(candidate)

        detections = remaining

    return selected


def decode(output, output_scale, output_zero_point):
    output_float = (
        output.astype(np.float32) - output_zero_point
    ) * output_scale

    detections = []

    point = 0

    for stride in STRIDES:
        grid_size = IMAGE_SIZE // stride

        for row in range(grid_size):
            for col in range(grid_size):

                values = output_float[0, point]

                class_logits = values[:NUM_CLASSES]

                class_scores = sigmoid(
                    class_logits
                )

                class_id = int(
                    np.argmax(class_scores)
                )

                score = float(
                    class_scores[class_id]
                )

                if score >= SCORE_THRESHOLD:

                    regression = values[
                        NUM_CLASSES:
                    ]

                    distances = distribution_integral(
                        regression
                    )

                    center_x = (
                        col + 0.5
                    ) * stride

                    center_y = (
                        row + 0.5
                    ) * stride

                    left = (
                        center_x
                        - distances[0] * stride
                    )

                    top = (
                        center_y
                        - distances[1] * stride
                    )

                    right = (
                        center_x
                        + distances[2] * stride
                    )

                    bottom = (
                        center_y
                        + distances[3] * stride
                    )

                    box = (
                        float(left),
                        float(top),
                        float(right),
                        float(bottom)
                    )

                    detections.append(
                        {
                            "point": point,
                            "stride": stride,
                            "row": row,
                            "col": col,
                            "class_id": class_id,
                            "score": score,
                            "box": box
                        }
                    )

                point += 1

    return apply_nms(detections)


def main():
    print("=" * 80)
    print("NanoDet INT8 CAMERA MODEL ANALYSIS")
    print("=" * 80)

    print()
    print("MODEL:")
    print(MODEL)

    interpreter = Interpreter(
        model_path=MODEL,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF
    )

    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    input_scale, input_zero_point = (
        input_info["quantization"]
    )

    output_scale, output_zero_point = (
        output_info["quantization"]
    )

    print()
    print("INPUT")
    print("shape:", input_info["shape"])
    print("dtype:", input_info["dtype"])
    print(
        "quantization:",
        input_scale,
        input_zero_point
    )

    print()
    print("OUTPUT")
    print("shape:", output_info["shape"])
    print("dtype:", output_info["dtype"])
    print(
        "quantization:",
        output_scale,
        output_zero_point
    )

    image_files = []

    for root, _, files in os.walk(DATASET_DIR):
        for filename in files:
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                image_files.append(
                    os.path.join(root, filename)
                )

    image_files = sorted(image_files)[:5]

    print()
    print("IMAGES:")
    for image_path in image_files:
        print(
            " ",
            os.path.basename(image_path)
        )

    all_outputs = []
    all_detections = []

    print()
    print("=" * 80)

    for image_path in image_files:

        image_name = os.path.basename(
            image_path
        )

        input_data = preprocess(
            image_path,
            input_scale,
            input_zero_point
        )

        interpreter.set_tensor(
            input_info["index"],
            input_data[np.newaxis, ...]
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_info["index"]
        ).copy()

        detections = decode(
            output,
            output_scale,
            output_zero_point
        )

        all_outputs.append(output)
        all_detections.append(detections)

        print()
        print(image_name)
        print("-" * 80)

        print(
            "INPUT:",
            "min=",
            int(input_data.min()),
            "max=",
            int(input_data.max()),
            "unique=",
            len(np.unique(input_data)),
            "mean=",
            round(
                float(input_data.mean()),
                3
            )
        )

        print(
            "OUTPUT:",
            "min=",
            int(output.min()),
            "max=",
            int(output.max()),
            "unique=",
            len(np.unique(output))
        )

        print()
        print("DETECTIONS:", len(detections))

        for i, detection in enumerate(detections):

            x1, y1, x2, y2 = (
                detection["box"]
            )

            class_name = CLASS_NAMES[
                detection["class_id"]
            ]

            print(
                f"DET {i}: "
                f"{class_name} "
                f"score={detection['score']:.4f} "
                f"point={detection['point']} "
                f"stride={detection['stride']} "
                f"grid=({detection['row']},{detection['col']}) "
                f"box=("
                f"{x1:.1f},"
                f"{y1:.1f},"
                f"{x2:.1f},"
                f"{y2:.1f}"
                f")"
            )

        print()
        print("TOP RAW OUTPUT POINTS:")

        raw = output[0].astype(np.float32)

        raw_scores = []

        for point in range(raw.shape[0]):

            values = (
                raw[point]
                - output_zero_point
            ) * output_scale

            class_scores = sigmoid(
                values[:NUM_CLASSES]
            )

            class_id = int(
                np.argmax(class_scores)
            )

            score = float(
                class_scores[class_id]
            )

            raw_scores.append(
                (
                    score,
                    point,
                    class_id,
                    values
                )
            )

        raw_scores.sort(
            key=lambda x: x[0],
            reverse=True
        )

        for score, point, class_id, values in raw_scores[:10]:

            print(
                f"point={point} "
                f"class={CLASS_NAMES[class_id]} "
                f"score={score:.4f}"
            )

            print(
                "  output:",
                np.round(values, 3)
            )

    print()
    print("=" * 80)
    print("CROSS-IMAGE COMPARISON")
    print("=" * 80)

    identical = True

    for i in range(1, len(all_outputs)):

        if not np.array_equal(
            all_outputs[0],
            all_outputs[i]
        ):
            identical = False
            break

    print()
    print(
        "Entire output identical:",
        identical
    )

    print()

    for i in range(1, len(all_outputs)):

        difference = np.abs(
            all_outputs[0].astype(np.int16)
            - all_outputs[i].astype(np.int16)
        )

        print(
            f"Output difference image 0 -> image {i}: "
            f"max={difference.max()} "
            f"mean={difference.mean():.4f} "
            f"changed_elements="
            f"{np.count_nonzero(difference)}"
        )

    print()
    print("DETECTION COMPARISON")

    for i, detections in enumerate(
        all_detections
    ):

        print(
            f"Image {i}: "
            f"{len(detections)} detections"
        )

        for detection in detections:

            x1, y1, x2, y2 = (
                detection["box"]
            )

            print(
                f"  "
                f"{CLASS_NAMES[detection['class_id']]} "
                f"{detection['score']:.4f} "
                f"("
                f"{x1:.1f},"
                f"{y1:.1f},"
                f"{x2:.1f},"
                f"{y2:.1f}"
                f")"
            )


if __name__ == "__main__":
    main()