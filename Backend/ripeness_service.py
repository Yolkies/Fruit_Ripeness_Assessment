import cv2
from pathlib import Path
from functools import lru_cache

from ultralytics import YOLO

import preprocessing

from fruit_pipeline_gui import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    segment,
    contour_count,
    annotate,
)

from fruit_ripeness.apple_ripeness.apple_ripeness_yolo import (
    assess_apple_ripeness,
)

from fruit_ripeness.banana_ripeness.banana_ripeness_yolo import (
    assess_banana_ripeness,
)

from fruit_ripeness.mango_ripeness.mango_ripeness_yolo import (
    assess_mango_ripeness,
)

from fruit_ripeness.orange_ripeness.orange_ripeness_yolo import (
    assess_orange_ripeness,
)

from fruit_ripeness.strawberry_ripeness.strawberry_ripeness_yolo import (
    assess_strawberry_ripeness,
)

from fruit_ripeness.tomato_ripeness.tomato_ripeness_yolo import (
    assess_tomato_ripeness,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "best4.pt"


@lru_cache(maxsize=1)
def load_model():
    """
    Load YOLO only once instead of reloading it every time
    Streamlit reruns the script.
    """
    return YOLO(str(MODEL_PATH))


def run_yolo_segmentation(image_bgr, model):
    """
    Same preprocessing + YOLO segmentation flow used by the
    existing test_*_full_pipeline.py files.
    """

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


def crop_detection_from_original(
    image_bgr,
    detection,
):
    """
    Same crop logic used by your full-pipeline test files.

    Returns:
        original_crop
        mask_crop
        background_removed
    """

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
        x:x + w
    ].copy()

    mask_crop = mask_full[
        y:y + h,
        x:x + w
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


def assess_single_fruit(
    fruit_type,
    original_crop,
    fruit_mask,
):
    """
    Send a detected fruit to its existing ripeness algorithm.
    """

    fruit_type = fruit_type.lower()

    if fruit_type == "apple":

        result, diagnostics = assess_apple_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
        )

    elif fruit_type == "banana":

        result, diagnostics = assess_banana_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
        )

    elif fruit_type == "mango":

        result, diagnostics = assess_mango_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
        )

    elif fruit_type == "orange":

        result, diagnostics = assess_orange_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
        )

    elif fruit_type == "strawberry":

        result, diagnostics = assess_strawberry_ripeness(
            original_crop,
            fruit_mask=fruit_mask,

            # Your strawberry code already supports this argument.
            # Change to False if you want strict mode.
            use_supplementary_index1_rule=True,
        )

    elif fruit_type == "tomato":

        result, diagnostics = assess_tomato_ripeness(
            original_crop,
            fruit_mask=fruit_mask,
        )

    else:
        raise ValueError(
            f"Unsupported fruit type: {fruit_type}"
        )

    return result, diagnostics


def normalise_result(
    fruit_type,
    result,
):
    """
    Convert the six different result dataclasses into one
    common format for Streamlit.
    """

    fruit_type = fruit_type.lower()

    if fruit_type == "apple":

        return {
            "fruit_type": "Apple",
            "ripeness_index": result.maturity_index,
            "ripeness_label": result.maturity_label,
            "rule_basis": result.rule_basis,
            "details": {
                "red_percentage": result.red_pct,
                "yellow_percentage": result.yellow_pct,
                "other_percentage": result.other_pct,
            },
        }

    elif fruit_type == "banana":

        return {
            "fruit_type": "Banana",
            "ripeness_index": result.ripeness_index,
            "ripeness_label": result.ripeness_label,
            "rule_basis": result.rule_basis,
            "details": {
                "yellow_percentage": result.yellow_pct,
                "brown_percentage": result.brown_pct,
                "developed_peel_percentage":
                    result.developed_peel_pct,
                "other_percentage": result.other_pct,
            },
        }

    elif fruit_type == "mango":

        return {
            "fruit_type": "Mango",
            "ripeness_index": result.maturity_stage,
            "ripeness_label": result.maturity_label,
            "rule_basis": result.rule_basis,
            "details": {
                "yellow_percentage": result.yellow_pct,
                "other_percentage": result.other_pct,
            },
        }

    elif fruit_type == "orange":

        return {
            "fruit_type": "Orange",
            "ripeness_index": result.maturity_index,
            "ripeness_label": result.maturity_label,
            "rule_basis": result.rule_basis,
            "details": {
                "yellow_orange_percentage":
                    result.yellow_orange_pct,
                "non_yellow_orange_percentage":
                    result.non_yellow_orange_pct,
                "mean_cci":
                    result.mean_cci,
                "median_cci":
                    result.median_cci,
            },
        }

    elif fruit_type == "strawberry":

        return {
            "fruit_type": "Strawberry",
            "ripeness_index": result.maturity_index,
            "ripeness_label": result.maturity_label,
            "rule_basis": result.rule_basis,
            "details": {
                "red_percentage":
                    result.red_pct,
                "normal_red_percentage":
                    result.normal_red_pct,
                "dark_red_percentage":
                    result.dark_red_pct,
                "other_percentage":
                    result.other_pct,
            },
        }

    elif fruit_type == "tomato":

        return {
            "fruit_type": "Tomato",
            "ripeness_index": result.maturity_index,
            "ripeness_label": result.maturity_label,
            "rule_basis": result.rule_basis,
            "details": {
                "green_percentage":
                    result.green_pct,
                "turning_percentage":
                    result.turning_pct,
                "pink_percentage":
                    result.pink_pct,
                "mature_red_percentage":
                    result.mature_red_pct,
                "colour_break_percentage":
                    result.colour_break_pct,
                "red_orange_percentage":
                    result.red_orange_pct,
            },
        }

    raise ValueError(
        f"Unsupported fruit type: {fruit_type}"
    )


def analyze_image(image_bgr):

    """
    COMPLETE PIPELINE FOR STREAMLIT

    image
        -> preprocessing
        -> YOLO segmentation
        -> fruit identification
        -> ripeness assessment
        -> Streamlit-friendly results
    """

    if image_bgr is None:
        raise ValueError("Input image is empty.")

    model = load_model()

    detections = run_yolo_segmentation(
        image_bgr,
        model,
    )

    if not detections:
        return {
            "count": 0,
            "annotated_image": image_bgr,
            "fruits": [],
        }

    annotated_image = annotate(
        image_bgr,
        detections,
        show_mask=True,
    )

    fruits = []

    for i, detection in enumerate(
        detections,
        start=1,
    ):

        class_id = detection["cls"]

        fruit_type = CLASS_NAMES[class_id]

        (
            original_crop,
            fruit_mask,
            background_removed,
        ) = crop_detection_from_original(
            image_bgr,
            detection,
        )

        result, diagnostics = assess_single_fruit(
            fruit_type,
            original_crop,
            fruit_mask,
        )

        normalised = normalise_result(
            fruit_type,
            result,
        )

        normalised["number"] = i
        normalised["confidence"] = detection["conf"]
        normalised["crop"] = original_crop
        normalised["masked_crop"] = background_removed
        normalised["diagnostics"] = diagnostics

        fruits.append(normalised)

    return {
        "count": len(fruits),
        "annotated_image": annotated_image,
        "fruits": fruits,
    }



def assess_detected_fruit(
    detected_fruit,
    selected_fruit_type=None,
):
    """
    Step 2:
    Run ripeness assessment after the user confirms
    or overrides YOLO's fruit prediction.
    """

    yolo_fruit_type = detected_fruit[
        "yolo_fruit_type"
    ]

    # If user does not override,
    # use YOLO's predicted fruit
    if selected_fruit_type is None:
        fruit_type = yolo_fruit_type
    else:
        fruit_type = (
            selected_fruit_type
            .lower()
            .strip()
        )

    original_crop = detected_fruit[
        "crop"
    ]

    fruit_mask = detected_fruit[
        "fruit_mask"
    ]

    # Run selected fruit's existing ripeness code
    result, diagnostics = assess_single_fruit(
        fruit_type,
        original_crop,
        fruit_mask,
    )

    # Convert result into common UI format
    normalised = normalise_result(
        fruit_type,
        result,
    )

    normalised["number"] = (
        detected_fruit["number"]
    )

    normalised["confidence"] = (
        detected_fruit["confidence"]
    )

    normalised["yolo_fruit_type"] = (
        yolo_fruit_type
    )

    normalised["selected_fruit_type"] = (
        fruit_type
    )

    normalised["crop"] = (
        original_crop
    )

    normalised["masked_crop"] = (
        detected_fruit["masked_crop"]
    )

    normalised["diagnostics"] = (
        diagnostics
    )

    return normalised


def detect_image(image_bgr):
    """
    Step 1:
    Only perform preprocessing + YOLO fruit detection.

    Ripeness analysis is NOT performed yet.
    """

    if image_bgr is None:
        raise ValueError("Input image is empty.")

    model = load_model()

    detections = run_yolo_segmentation(
        image_bgr,
        model,
    )

    if not detections:
        return {
            "count": 0,
            "annotated_image": image_bgr,
            "detections": [],
        }

    annotated_image = annotate(
        image_bgr,
        detections,
        show_mask=True,
    )

    prepared_detections = []

    for i, detection in enumerate(
        detections,
        start=1,
    ):

        class_id = int(detection["cls"])

        yolo_fruit_type = CLASS_NAMES[
            class_id
        ]

        (
            original_crop,
            fruit_mask,
            background_removed,
        ) = crop_detection_from_original(
            image_bgr,
            detection,
        )

        prepared_detections.append(
            {
                "number": i,

                # What YOLO predicted
                "yolo_fruit_type":
                    yolo_fruit_type,

                "confidence":
                    float(detection["conf"]),

                # Keep these for later ripeness analysis
                "crop":
                    original_crop,

                "fruit_mask":
                    fruit_mask,

                "masked_crop":
                    background_removed,
            }
        )

    return {
        "count":
            len(prepared_detections),

        "annotated_image":
            annotated_image,

        "detections":
            prepared_detections,
    }



