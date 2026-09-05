"""
ORANGE FULL PIPELINE TEST

Flow:
    original image
        -> teammate preprocessing
        -> teammate YOLO segmentation
        -> YOLO orange mask
        -> original-colour orange crop
        -> CIELAB
        -> per-pixel CCI
        -> yellow/orange coverage
        -> citrus maturity Index 1-5

Debug structure:
orange_full_test_debug/
`-- orange_001.jpg/
    |-- 00_original_input.png
    |-- 01_yolo_segmented_full.png
    |-- 02_original_crop.png
    |-- 03_yolo_mask.png
    |-- 04_yolo_background_removed.png
    |-- result.txt
    `-- ripeness/
        |-- input_original_colour.png
        |-- fruit_mask.png
        |-- yellow_orange_mask.png
        |-- non_yellow_orange_mask.png
        |-- undefined_cci_mask.png
        |-- cci_visualization.png
        |-- lab_l_channel.png
        |-- lab_a_channel.png
        `-- lab_b_channel.png

Example:
    python fruit_ripeness\\orange_ripeness\\test_orange_full_pipeline.py "C:\\path\\orange_001.jpg" --model best4.pt
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from ultralytics import YOLO

import preprocessing

from detection import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    segment,
    contour_count,
    annotate,
)

from orange_ripeness_yolo import (
    assess_orange_ripeness,
    save_diagnostics,
)


def run_yolo_segmentation(image_bgr, model):
    """Reuse teammate preprocessing + YOLO segmentation."""
    prep = preprocessing.preprocess(image_bgr)

    if not prep.crops:
        crops = [prep.bgr]
        crop_rects = [
            (0, 0, prep.bgr.shape[1], prep.bgr.shape[0])
        ]
    else:
        crops = prep.crops
        crop_rects = prep.crop_rects

    detections = segment(
        model,
        crops,
        crop_rects,
        image_bgr.shape,
        conf=CONF_THRESHOLD,
    )

    contour_count(detections)

    return detections


def crop_detection_from_original(image_bgr, detection):
    """Crop original-colour orange and corresponding YOLO mask."""
    mask_full = detection["mask"]

    if detection.get("contour") is not None:
        x, y, w, h = cv2.boundingRect(
            detection["contour"]
        )
    else:
        x, y, w, h = cv2.boundingRect(
            mask_full
        )

    original_crop = image_bgr[
        y:y + h,
        x:x + w,
    ].copy()

    mask_crop = mask_full[
        y:y + h,
        x:x + w,
    ].copy()

    background_removed = cv2.bitwise_and(
        original_crop,
        original_crop,
        mask=mask_crop,
    )

    return (
        original_crop,
        mask_crop,
        background_removed,
    )


def get_unique_folder_name(output_root, base_name):
    """Use input filename; append _2, _3, ... if it already exists."""
    candidate = os.path.join(
        output_root,
        base_name,
    )

    if not os.path.exists(candidate):
        return candidate

    counter = 2

    while True:
        candidate = os.path.join(
            output_root,
            f"{base_name}_{counter}",
        )

        if not os.path.exists(candidate):
            return candidate

        counter += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help=r'Input image, e.g. "C:\...\orange_001.jpg"',
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Teammate's final YOLO model, e.g. best4.pt",
    )

    parser.add_argument(
        "--out",
        default=str(
            SCRIPT_DIR / "orange_full_test_debug"
        ),
        help="Persistent debug folder. Existing results are kept.",
    )

    args = parser.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        raise SystemExit(
            f"Cannot read input image: {args.image}"
        )

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Cannot find model: {args.model}"
        )

    os.makedirs(
        args.out,
        exist_ok=True,
    )

    input_filename = Path(args.image).name

    fruit_dir = get_unique_folder_name(
        args.out,
        input_filename,
    )

    ripeness_dir = os.path.join(
        fruit_dir,
        "ripeness",
    )

    os.makedirs(
        ripeness_dir,
        exist_ok=False,
    )

    print("Loading YOLO model...")

    model = YOLO(args.model)

    detections = run_yolo_segmentation(
        image,
        model,
    )

    print(
        f"Total detected fruits: {len(detections)}"
    )

    yolo_visual = annotate(
        image,
        detections,
        show_mask=True,
    )

    cv2.imwrite(
        os.path.join(
            fruit_dir,
            "00_original_input.png",
        ),
        image,
    )

    cv2.imwrite(
        os.path.join(
            fruit_dir,
            "01_yolo_segmented_full.png",
        ),
        yolo_visual,
    )

    orange_detections = []

    for d in detections:
        fruit_type = CLASS_NAMES[d["cls"]]

        if fruit_type.lower() == "orange":
            orange_detections.append(d)

    if not orange_detections:
        print("\nNo orange detected.")
        print(
            "Folder still created for inspection:",
            os.path.abspath(fruit_dir),
        )
        return

    # Same behavior as strawberry/tomato/apple testers:
    # use the first detected orange.
    d = orange_detections[0]

    (
        original_crop,
        yolo_mask,
        background_removed,
    ) = crop_detection_from_original(
        image,
        d,
    )

    cv2.imwrite(
        os.path.join(
            fruit_dir,
            "02_original_crop.png",
        ),
        original_crop,
    )

    cv2.imwrite(
        os.path.join(
            fruit_dir,
            "03_yolo_mask.png",
        ),
        yolo_mask,
    )

    cv2.imwrite(
        os.path.join(
            fruit_dir,
            "04_yolo_background_removed.png",
        ),
        background_removed,
    )

    result, diagnostics = assess_orange_ripeness(
        original_crop,
        fruit_mask=yolo_mask,
    )

    save_diagnostics(
        ripeness_dir,
        diagnostics,
    )

    mean_cci_text = (
        "N/A"
        if result.mean_cci is None
        else f"{result.mean_cci:.2f}"
    )

    median_cci_text = (
        "N/A"
        if result.median_cci is None
        else f"{result.median_cci:.2f}"
    )

    result_lines = [
        "=" * 70,
        f"INPUT FILE             : {args.image}",
        f"FOLDER NAME            : {os.path.basename(fruit_dir)}",
        "=" * 70,
        f"YOLO confidence        : {d['conf']:.4f}",
        f"YOLO ROI pixels        : {result.total_fruit_pixels}",
        "",
        "PIXEL COUNTS",
        (
            f"Yellow/orange pixels   : "
            f"{result.yellow_orange_pixels} (CCI >= 0)"
        ),
        (
            f"Non-yellow/orange pixels: "
            f"{result.non_yellow_orange_pixels} (CCI < 0)"
        ),
        (
            f"Undefined CCI pixels   : "
            f"{result.undefined_cci_pixels}"
        ),
        "",
        "COVERAGE",
        (
            f"Yellow/orange coverage : "
            f"{result.yellow_orange_pct:.2f}%"
        ),
        (
            f"Non-yellow/orange cov. : "
            f"{result.non_yellow_orange_pct:.2f}%"
        ),
        (
            f"Undefined CCI coverage : "
            f"{result.undefined_cci_pct:.2f}%"
        ),
        "",
        "CCI SUMMARY",
        f"Mean CCI               : {mean_cci_text}",
        f"Median CCI             : {median_cci_text}",
        "",
        f"Prediction             : {result.maturity_label}",
        f"Rule basis             : {result.rule_basis}",
        f"Saved to               : {os.path.abspath(fruit_dir)}",
    ]

    result_text = "\n".join(result_lines)

    result_txt_path = os.path.join(
        fruit_dir,
        "result.txt",
    )

    with open(
        result_txt_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(result_text)
        file.write("\n")

    print("\n" + result_text)

    print(
        f"Result text saved to   : "
        f"{os.path.abspath(result_txt_path)}"
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()
