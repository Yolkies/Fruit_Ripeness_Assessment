"""
TOMATO FULL PIPELINE TEST

Purpose
-------
Test one tomato image through the current complete flow:

    original image
        -> teammate preprocessing
        -> teammate YOLO segmentation
        -> YOLO tomato mask / background removed
        -> tomato_ripeness_yolo.py
        -> HSV percentages + CIELAB L*
        -> tomato maturity result

Project folder layout
---------------------
Fruit_Ripeness_Assessment/
|
|-- preprocessing.py
|-- detection.py
|-- best4.pt                         # example; use teammate's final model
|
`-- fruit_ripeness/tomato_ripeness/
    |-- tomato_ripeness_yolo.py
    |-- test_tomato_full_pipeline.py
    |
    `-- tomato_full_test_debug/
        `-- tomato_001.jpg/
            |-- 00_original_input.png
            |-- 01_yolo_segmented_full.png
            |-- 02_original_crop.png
            |-- 03_yolo_mask.png
            |-- 04_yolo_background_removed.png
            |-- result.txt
            |
            `-- ripeness/
                |-- input_original_colour.png
                |-- fruit_mask.png
                |-- green_mask.png
                |-- mature_red_mask.png
                |-- colour_break_mask.png
                |-- red_orange_mask.png
                |-- other_mask.png
                `-- lab_l_channel.png

Run from Fruit_Ripeness_Assessment project root
-----------------------------------------------
Example:

    python fruit_ripeness\\tomato_ripeness\\test_tomato_full_pipeline.py "C:\\path\\to\\tomato_001.jpg" --model best4.pt

Results are stored inside:

    fruit_ripeness\\tomato_ripeness\\tomato_full_test_debug\\tomato_001.jpg\\

The same terminal result is also saved as:

    fruit_ripeness\\tomato_ripeness\\tomato_full_test_debug\\tomato_001.jpg\\result.txt

If the same input file name is tested again, the previous result is kept:

    tomato_001.jpg_2
    tomato_001.jpg_3
    ...

Important
---------
- The teammate YOLO mask is used directly.
- No GrabCut is used.
- Original-colour pixels are used for ripeness analysis.
- Index 1 and Index 2 are combined for now.
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

from tomato_ripeness_yolo import (
    assess_tomato_ripeness,
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
    """Crop original-colour tomato and corresponding YOLO mask."""
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
        help=r'Input image, e.g. "C:\...\tomato_001.jpg"',
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Teammate's final YOLO segmentation model, e.g. best4.pt",
    )

    parser.add_argument(
        "--out",
        default=str(
            SCRIPT_DIR / "tomato_full_test_debug"
        ),
        help=(
            "Persistent debug folder. Existing results are kept."
        ),
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

    tomato_detections = []

    for d in detections:
        fruit_type = CLASS_NAMES[
            d["cls"]
        ]

        if fruit_type.lower() == "tomato":
            tomato_detections.append(d)

    if not tomato_detections:
        print("\nNo tomato detected.")
        print(
            "Folder still created for inspection:",
            os.path.abspath(fruit_dir),
        )
        return

    # Same simple behavior as the strawberry tester:
    # use the first detected tomato.
    d = tomato_detections[0]

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

    result, diagnostics = assess_tomato_ripeness(
        original_crop,
        fruit_mask=yolo_mask,
    )

    save_diagnostics(
        ripeness_dir,
        diagnostics,
    )

    mean_lstar_text = (
        "N/A"
        if result.mean_lstar_red_orange is None
        else f"{result.mean_lstar_red_orange:.2f}"
    )

    median_lstar_text = (
        "N/A"
        if result.median_lstar_red_orange is None
        else f"{result.median_lstar_red_orange:.2f}"
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
        f"Green pixels           : {result.green_pixels}",
        f"Turning pixels         : {result.turning_pixels}",
        f"Pink pixels            : {result.pink_pixels}",
        f"Mature/red pixels      : {result.mature_red_pixels}",
        f"Colour-break pixels    : {result.colour_break_pixels} (Turning + Pink + Mature/red)",
        f"Red-orange pixels      : {result.red_orange_pixels} (Pink + Mature/red)",
        "",
        "COVERAGE",
        f"Green coverage         : {result.green_pct:.2f}%",
        f"Colour-break coverage  : {result.colour_break_pct:.2f}%",
        f"Red-orange coverage    : {result.red_orange_pct:.2f}%",
        f"Other coverage         : {result.other_pct:.2f}%",
        "",
        "INDEX 5 / 6 L* CHECK",
        f"Mean L* mature-red     : {mean_lstar_text}",
        f"Median L* mature-red   : {median_lstar_text}",
        f"L* Index 5/6 threshold : 40.70",
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
