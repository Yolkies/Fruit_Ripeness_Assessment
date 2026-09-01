"""
BANANA FULL PIPELINE TEST - YELLOW + BROWN VERSION

Flow
----
original image
 -> teammate preprocessing
 -> teammate YOLO segmentation
 -> original-colour banana crop + YOLO mask
 -> banana_ripeness_yolo.py
 -> brown mask FIRST
 -> yellow mask
 -> yellow/brown coverage
 -> Index 1-7

Decision order
--------------
1. Brown >= 20% -> Index 7 - Overripe
2. Small brown spotting on an otherwise developed yellow banana -> Index 7
3. Otherwise use yellow coverage:
       <= 1%          -> Index 1
       < 30%          -> Index 2
       30% to < 50%   -> Index 3
       50% to 80%     -> Index 4
       > 80% to < 95% -> Index 5
       >= 95%         -> Index 6

Reference for brown-area reasoning
----------------------------------
Mendoza, F. & Aguilera, J. M. (2004),
"Application of Image Analysis for Classification of Ripening Bananas",
Journal of Food Science, 69(9), E471-E477.
DOI: 10.1111/j.1365-2621.2004.tb09932.x

The paper supports brown-area percentage (%BSA) as a useful banana ripening
feature and reports late-ripening values reaching about 20-50%. The 20% cutoff
used here is a project operational threshold motivated by that observation,
not an official threshold stated by the paper.

Run from project root
---------------------
python banana_ripeness\\test_banana_full_pipeline.py "C:\\path\\banana.jpg" --model best4.pt

Debug output
------------
banana_full_test_debug/
`-- banana.jpg/
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
        |-- yellow_mask.png
        |-- brown_mask.png
        |-- developed_peel_mask.png
        `-- other_mask.png
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )
    
import cv2
from ultralytics import YOLO

import preprocessing

from fruit_pipeline_gui import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    segment,
    contour_count,
    annotate,
)

from banana_ripeness_yolo import (
    assess_banana_ripeness,
    save_diagnostics,
)


def run_yolo_segmentation(image_bgr, model):
    """Reuse teammate preprocessing + YOLO segmentation."""
    prep = preprocessing.preprocess(image_bgr)

    if not prep.crops:
        crops = [prep.bgr]
        crop_rects = [
            (
                0,
                0,
                prep.bgr.shape[1],
                prep.bgr.shape[0],
            )
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
    """Crop original-colour banana and its YOLO mask."""
    mask_full = detection["mask"]

    if detection.get("contour") is not None:
        x, y, w, h = cv2.boundingRect(detection["contour"])
    else:
        x, y, w, h = cv2.boundingRect(mask_full)

    original_crop = image_bgr[y:y + h, x:x + w].copy()
    mask_crop = mask_full[y:y + h, x:x + w].copy()

    background_removed = cv2.bitwise_and(
        original_crop,
        original_crop,
        mask=mask_crop,
    )

    return original_crop, mask_crop, background_removed


def get_unique_folder_name(output_root, base_name):
    """Use filename, then append _2, _3, ... for repeated tests."""
    candidate = os.path.join(output_root, base_name)

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
        help="Input banana image",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="YOLO segmentation model, e.g. best4.pt",
    )

    parser.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "banana_full_test_debug"),
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

    os.makedirs(args.out, exist_ok=True)

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

    banana_detections = []

    for d in detections:
        fruit_type = CLASS_NAMES[d["cls"]]

        if fruit_type.lower() == "banana":
            banana_detections.append(d)

    if not banana_detections:
        print("\nNo banana detected.")
        print(
            "Folder still created for inspection:",
            os.path.abspath(fruit_dir),
        )
        return

    # Keep the same tester behaviour: use the first detected banana.
    d = banana_detections[0]

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

    result, diagnostics = assess_banana_ripeness(
        original_crop,
        fruit_mask=yolo_mask,
    )

    save_diagnostics(
        ripeness_dir,
        diagnostics,
    )

    result_lines = [
        "=" * 74,
        f"INPUT FILE                  : {args.image}",
        f"FOLDER NAME                 : {os.path.basename(fruit_dir)}",
        "=" * 74,
        f"YOLO confidence             : {d['conf']:.4f}",
        f"YOLO ROI pixels             : {result.total_fruit_pixels}",
        "",
        "PIXEL RULES",
        "Yellow                     : OpenCV H 20-35, S 40-255, V 80-255",
        "Brown                      : OpenCV H 0-25, S 64-255, V 0-153",
        "",
        "DECISION RULE ORDER",
        "Brown >= 20%               : Index 7 - Overripe (checked FIRST)",
        "Small brown on yellow      : Index 7 - Brown-spotted",
        "Yellow <= 1%               : Index 1",
        "Yellow < 30%               : Index 2",
        "30% <= Yellow < 50%        : Index 3",
        "50% <= Yellow <= 80%       : Index 4",
        "80% < Yellow < 95%         : Index 5",
        "Yellow >= 95%              : Index 6",
        "",
        "PIXEL COUNTS",
        f"Yellow pixels                : {result.yellow_pixels}",
        f"Brown pixels                 : {result.brown_pixels}",
        f"Developed peel pixels        : {result.developed_peel_pixels}",
        f"Other pixels                 : {result.other_pixels}",
        "",
        "COVERAGE",
        f"Yellow coverage              : {result.yellow_pct:.2f}%",
        f"Brown coverage               : {result.brown_pct:.2f}%",
        f"Developed peel coverage      : {result.developed_peel_pct:.2f}%",
        f"Other coverage               : {result.other_pct:.2f}%",
        "",
        f"Ripeness Factor (RF)         : {result.ripeness_factor:.6f}",
        "",
        f"Prediction                   : {result.ripeness_label}",
        f"Rule basis                   : {result.rule_basis}",
        f"Saved to                     : {os.path.abspath(fruit_dir)}",
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
        file.write(result_text + "\n")

    print("\n" + result_text)

    print(
        f"Result text saved to        : "
        f"{os.path.abspath(result_txt_path)}"
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()
