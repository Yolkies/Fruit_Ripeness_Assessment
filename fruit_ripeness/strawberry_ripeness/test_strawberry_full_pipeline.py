"""
STRAWBERRY FULL PIPELINE TEST

Purpose
-------
Test one strawberry image through the complete current flow:

    original image
        -> teammate preprocessing
        -> teammate YOLO segmentation
        -> YOLO strawberry mask / background removed
        -> strawberry_ripeness_yolo.py
        -> red / dark-red percentage
        -> ripeness result

Project folder layout
---------------------
Fruit_Ripeness_Assessment/
|
|-- preprocessing.py
|-- fruit_pipeline_gui.py
|-- best4.pt                         # example; use teammate's final model
|
`-- fruit_ripeness/strawberry_ripeness/
    |-- strawberry_ripeness_yolo.py
    |-- test_strawberry_full_pipeline.py
    |
    `-- strawberry_full_test_debug/
        `-- fresa_033.jpg/
            |-- 00_original_input.png
            |-- 01_yolo_segmented_full.png
            |-- 02_original_crop.png
            |-- 03_yolo_mask.png
            |-- 04_yolo_background_removed.png
            |
            `-- ripeness/
                |-- input_original_colour.png
                |-- fruit_mask.png
                |-- red_mask.png
                `-- dark_red_mask.png

Run from the Fruit_Ripeness_Assessment project root
---------------------------------------------------
Example:

    python fruit_ripeness\\strawberry_ripeness\\test_strawberry_full_pipeline.py "C:\\Users\\Lenovo\\OneDrive\\Desktop\\UNI\\UNI_ASSIGNMENT\\Fruit_Ripeness_Assessment\\dataset\\stawberry\\strawberryDataset\\Pickable\\fresa_033.jpg" --model best4.pt

The result is automatically saved inside:

    fruit_ripeness\\strawberry_ripeness\\strawberry_full_test_debug\\fresa_033.jpg\\

If another input file is tested, its own original file name is used as the
folder name.

If the exact same file name is tested again, the previous result is kept and
a suffix is added:

    fresa_033.jpg_2
    fresa_033.jpg_3
    ...

Important
---------
- The YOLO mask from the teammate's segmentation is used directly.
- No GrabCut is used.
- No separate calyx-removal stage is used.
- Ripeness analysis uses the ORIGINAL-colour crop, not the CLAHE/preprocessed
  colour image.
"""


import argparse
import os
import sys
from pathlib import Path

# This file lives inside:
#   Fruit_Ripeness_Assessment/fruit_ripeness/strawberry_ripeness/
#
# Add the project root to sys.path so teammate modules two levels above
# can be imported reliably when this script is run from the project root.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

from strawberry_ripeness_yolo import (
    assess_strawberry_ripeness,
    save_diagnostics,
)


def run_yolo_segmentation(image_bgr, model):
    """Run the same preprocessing + YOLO segmentation flow as teammate GUI."""
    prep = preprocessing.preprocess(image_bgr)

    if not prep.crops:
        crops = [prep.bgr]
        crop_rects = [(0, 0, prep.bgr.shape[1], prep.bgr.shape[0])]
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
    """Crop original-colour pixels and the matching YOLO mask."""
    mask_full = detection["mask"]

    if detection.get("contour") is not None:
        x, y, w, h = cv2.boundingRect(detection["contour"])
    else:
        x, y, w, h = cv2.boundingRect(mask_full)

    original_crop = image_bgr[y:y+h, x:x+w].copy()
    mask_crop = mask_full[y:y+h, x:x+w].copy()

    background_removed = cv2.bitwise_and(
        original_crop,
        original_crop,
        mask=mask_crop,
    )

    return original_crop, mask_crop, background_removed


def get_unique_folder_name(output_root, base_name):
    """
    Use the input file name as folder name.
    If same name already exists, append _2, _3, ...
    """
    candidate = os.path.join(output_root, base_name)

    if not os.path.exists(candidate):
        return candidate

    counter = 2
    while True:
        candidate = os.path.join(output_root, f"{base_name}_{counter}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help=r'Input image, e.g. "C:\...\fresa_033.jpg"',
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Teammate's final YOLO segmentation model, e.g. best4.pt",
    )

    parser.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "strawberry_full_test_debug"),
        help=(
            "Persistent output folder. By default, results are stored inside "
            "fruit_ripeness/strawberry_ripeness/strawberry_full_test_debug/. "
            "Existing results are kept."
        ),
    )

    parser.add_argument(
        "--supplementary-index1",
        action="store_true",
        help="Enable supplementary <10%% red Index-1 rule",
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

    # Create root folder once if it does not exist
    os.makedirs(args.out, exist_ok=True)

    # Use the file name only, e.g. fresa_033.jpg
    input_filename = Path(args.image).name

    # Create a unique folder for this input filename
    fruit_dir = get_unique_folder_name(args.out, input_filename)
    ripeness_dir = os.path.join(fruit_dir, "ripeness")

    os.makedirs(ripeness_dir, exist_ok=False)

    print("Loading YOLO model...")
    model = YOLO(args.model)

    detections = run_yolo_segmentation(image, model)

    print(f"Total detected fruits: {len(detections)}")

    # Full-image YOLO visualisation
    yolo_visual = annotate(
        image,
        detections,
        show_mask=True,
    )

    # Save original and YOLO visualisation inside the file-name folder
    cv2.imwrite(
        os.path.join(fruit_dir, "00_original_input.png"),
        image,
    )

    cv2.imwrite(
        os.path.join(fruit_dir, "01_yolo_segmented_full.png"),
        yolo_visual,
    )

    # Keep only strawberry detections
    strawberry_detections = []

    for d in detections:
        fruit_type = CLASS_NAMES[d["cls"]]
        if fruit_type.lower() == "strawberry":
            strawberry_detections.append(d)

    if not strawberry_detections:
        print("\nNo strawberry detected.")
        print(f"Folder still created for inspection: {os.path.abspath(fruit_dir)}")
        return

    # If more than one strawberry is detected, use the first one.
    # (You can later extend this if needed.)
    d = strawberry_detections[0]

    original_crop, yolo_mask, background_removed = crop_detection_from_original(
        image,
        d,
    )

    cv2.imwrite(
        os.path.join(fruit_dir, "02_original_crop.png"),
        original_crop,
    )

    cv2.imwrite(
        os.path.join(fruit_dir, "03_yolo_mask.png"),
        yolo_mask,
    )

    cv2.imwrite(
        os.path.join(fruit_dir, "04_yolo_background_removed.png"),
        background_removed,
    )

    result, diagnostics = assess_strawberry_ripeness(
        original_crop,
        fruit_mask=yolo_mask,
        use_supplementary_index1_rule=args.supplementary_index1,
    )

    save_diagnostics(
        ripeness_dir,
        diagnostics,
    )

    print("\n" + "=" * 60)
    print(f"INPUT FILE            : {args.image}")
    print(f"FOLDER NAME           : {os.path.basename(fruit_dir)}")
    print("=" * 60)
    print(f"YOLO confidence       : {d['conf']:.4f}")
    print(f"YOLO ROI pixels       : {result.total_flesh_pixels}")
    print(f"Red pixels            : {result.red_pixels}")
    print(f"Dark-red pixels       : {result.dark_red_pixels}")
    print(f"Red coverage          : {result.red_pct:.2f}%")
    print(f"Normal-red coverage   : {result.normal_red_pct:.2f}%")
    print(f"Dark-red coverage     : {result.dark_red_pct:.2f}%")
    print(f"Other coverage        : {result.other_pct:.2f}%")
    print(f"Prediction            : {result.maturity_label}")
    print(f"Rule basis            : {result.rule_basis}")
    print(f"Saved to              : {os.path.abspath(fruit_dir)}")

    print("\nFinished.")


if __name__ == "__main__":
    main()
