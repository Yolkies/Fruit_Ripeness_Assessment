# pip install opencv-python numpy

"""
APPLE RIPENESS ANALYSIS - FUJI-BASED COLOUR METHOD
==================================================

Important dataset note
----------------------
The Kaggle Fruit Recognition dataset names the class only as "Apple A".
Its exact cultivar is not confirmed by the dataset source.

This module therefore treats Apple A as a Fuji-like red/yellow apple for
project testing and applies Fuji apple references. Do not state that
"Apple A is confirmed Fuji" in the report.

REFERENCE 1 - FUJI MATURITY BY RED SKIN COVERAGE
-------------------------------------------------
KR101038935B1:
"Maturity judgment method of Fuji apple and aging judgment color chart
of Fuji apple"

Fuji maturity stages based on skin coloration rate:

    Index 1 : red coverage < 30%
    Index 2 : 30% <= red coverage < 50%
    Index 3 : 50% <= red coverage < 60%
    Index 4 : 60% <= red coverage < 80%
    Index 5 : red coverage >= 80%

REFERENCE 2 - FUJI APPLE HSV HUE
--------------------------------
Hu et al. (2021), Horticulturae 7(9), 276:
"Infield Apple Detection and Grading Based on Multi-Feature Fusion"

The study used OpenCV HSV Hue (0-180 scale) and reported:

    Red:
        H = 0-15
        OR
        H = 175-180

    Yellow:
        H ~= 20-25

The paper calculates red area ratio as:

    red pixels / total apple pixels

The paper mainly uses Hue because Hue is less affected by illumination.
It does not provide a complete S/V threshold for the reported red/yellow
ranges, so this module follows the published Hue rule directly.

OpenCV uint8 Hue normally has values 0-179, so the upper red interval
is implemented as 175-179.

PIPELINE
--------
Teammate YOLO mask
    -> original-colour apple pixels
    -> OpenCV HSV
    -> red mask / yellow mask
    -> red coverage %
    -> Index 1-5

No CLAHE, white balance, or colour enhancement is applied to ripeness pixels.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Fuji apple Hue ranges from Hu et al. (2021)
# OpenCV Hue range: 0-179
# ---------------------------------------------------------------------------

RED_HUE_RANGES = (
    (0, 15),
    (175, 179),
)

YELLOW_HUE_RANGE = (
    20,
    25,
)


# ---------------------------------------------------------------------------
# Fuji maturity red-coverage boundaries from KR101038935B1
# ---------------------------------------------------------------------------

INDEX2_MIN = 30.0
INDEX3_MIN = 50.0
INDEX4_MIN = 60.0
INDEX5_MIN = 80.0


@dataclass
class AppleRipenessResult:
    maturity_index: int
    maturity_label: str
    rule_basis: str

    total_fruit_pixels: int

    red_pixels: int
    yellow_pixels: int
    other_pixels: int

    red_pct: float
    yellow_pct: float
    other_pct: float


def _normalise_binary_mask(
    mask: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Convert supplied YOLO mask to a 0/255 uint8 binary mask."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_BGR2GRAY,
        )

    if mask.shape[:2] != target_shape:
        mask = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return (
        (mask > 0).astype(np.uint8) * 255
    )


def fruit_mask_from_black_background(
    masked_crop_bgr: np.ndarray,
) -> np.ndarray:
    """
    Fallback only.

    Final integration should pass the teammate YOLO mask directly.
    """
    if (
        masked_crop_bgr is None
        or masked_crop_bgr.size == 0
    ):
        raise ValueError(
            "Input apple crop is empty"
        )

    non_black = np.any(
        masked_crop_bgr != 0,
        axis=2,
    )

    return (
        non_black.astype(np.uint8) * 255
    )


def prepare_apple_for_colour_analysis(
    original_crop_bgr: np.ndarray,
    fruit_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Prepare original-colour pixels for apple ripeness analysis.
    """
    if (
        original_crop_bgr is None
        or original_crop_bgr.size == 0
    ):
        raise ValueError(
            "Input apple crop is empty"
        )

    h, w = original_crop_bgr.shape[:2]

    if fruit_mask is None:
        fruit_mask = (
            fruit_mask_from_black_background(
                original_crop_bgr
            )
        )
    else:
        fruit_mask = _normalise_binary_mask(
            fruit_mask,
            (h, w),
        )

    bgr = original_crop_bgr.copy()

    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV,
    )

    return {
        "bgr": bgr,
        "hsv": hsv,
        "fruit_mask": fruit_mask,
    }


def _make_red_hue_mask(
    hue: np.ndarray,
) -> np.ndarray:
    """
    Red Hue:
        0-15 OR 175-179
    """
    low_red = cv2.inRange(
        hue,
        RED_HUE_RANGES[0][0],
        RED_HUE_RANGES[0][1],
    )

    high_red = cv2.inRange(
        hue,
        RED_HUE_RANGES[1][0],
        RED_HUE_RANGES[1][1],
    )

    return cv2.bitwise_or(
        low_red,
        high_red,
    )


def _make_yellow_hue_mask(
    hue: np.ndarray,
) -> np.ndarray:
    """
    Yellow Hue:
        approximately 20-25
    """
    return cv2.inRange(
        hue,
        YELLOW_HUE_RANGE[0],
        YELLOW_HUE_RANGE[1],
    )


def measure_apple_features(
    original_crop_bgr: np.ndarray,
    fruit_mask: Optional[np.ndarray] = None,
) -> Tuple[
    AppleRipenessResult,
    Dict[str, np.ndarray],
]:
    """
    Measure Fuji-style red/yellow colour coverage only inside
    the YOLO apple mask.
    """
    data = prepare_apple_for_colour_analysis(
        original_crop_bgr,
        fruit_mask=fruit_mask,
    )

    hsv = data["hsv"]
    roi_mask = data["fruit_mask"]

    total = int(
        cv2.countNonZero(
            roi_mask
        )
    )

    if total == 0:
        raise ValueError(
            "YOLO apple mask contains no fruit pixels"
        )

    hue = hsv[:, :, 0]

    # Published Hue masks.
    red_raw = _make_red_hue_mask(
        hue
    )

    yellow_raw = _make_yellow_hue_mask(
        hue
    )

    # Restrict all colour measurements to the YOLO apple ROI.
    red_mask = cv2.bitwise_and(
        red_raw,
        roi_mask,
    )

    yellow_mask = cv2.bitwise_and(
        yellow_raw,
        roi_mask,
    )

    # Red and yellow Hue ranges do not overlap.
    classified_mask = cv2.bitwise_or(
        red_mask,
        yellow_mask,
    )

    other_mask = cv2.bitwise_and(
        roi_mask,
        cv2.bitwise_not(
            classified_mask
        ),
    )

    # Pixel counts.
    red_pixels = int(
        cv2.countNonZero(
            red_mask
        )
    )

    yellow_pixels = int(
        cv2.countNonZero(
            yellow_mask
        )
    )

    other_pixels = int(
        cv2.countNonZero(
            other_mask
        )
    )

    # Percentages.
    red_pct = (
        100.0 * red_pixels / total
    )

    yellow_pct = (
        100.0 * yellow_pixels / total
    )

    other_pct = (
        100.0 * other_pixels / total
    )

    # Temporary result before maturity classification.
    result = AppleRipenessResult(
        maturity_index=0,
        maturity_label="Not classified",
        rule_basis="",
        total_fruit_pixels=total,
        red_pixels=red_pixels,
        yellow_pixels=yellow_pixels,
        other_pixels=other_pixels,
        red_pct=red_pct,
        yellow_pct=yellow_pct,
        other_pct=other_pct,
    )

    diagnostics = {
        "bgr": data["bgr"],
        "fruit_mask": roi_mask,
        "hue_channel": hue,
        "red_mask": red_mask,
        "yellow_mask": yellow_mask,
        "other_mask": other_mask,
    }

    return result, diagnostics


def classify_apple(
    measurement: AppleRipenessResult,
) -> AppleRipenessResult:
    """
    Apply Fuji apple red skin-coverage maturity stages.

    Index 1 : red < 30%
    Index 2 : 30% <= red < 50%
    Index 3 : 50% <= red < 60%
    Index 4 : 60% <= red < 80%
    Index 5 : red >= 80%
    """
    red = measurement.red_pct

    if red < INDEX2_MIN:
        measurement.maturity_index = 1
        measurement.maturity_label = (
            "Index 1 - Stage 1"
        )
        measurement.rule_basis = (
            f"Red coverage={red:.2f}% (<30%)."
        )

    elif red < INDEX3_MIN:
        measurement.maturity_index = 2
        measurement.maturity_label = (
            "Index 2 - Stage 2"
        )
        measurement.rule_basis = (
            f"Red coverage={red:.2f}% "
            "(>=30% and <50%)."
        )

    elif red < INDEX4_MIN:
        measurement.maturity_index = 3
        measurement.maturity_label = (
            "Index 3 - Stage 3"
        )
        measurement.rule_basis = (
            f"Red coverage={red:.2f}% "
            "(>=50% and <60%)."
        )

    elif red < INDEX5_MIN:
        measurement.maturity_index = 4
        measurement.maturity_label = (
            "Index 4 - Stage 4"
        )
        measurement.rule_basis = (
            f"Red coverage={red:.2f}% "
            "(>=60% and <80%)."
        )

    else:
        measurement.maturity_index = 5
        measurement.maturity_label = (
            "Index 5 - Stage 5"
        )
        measurement.rule_basis = (
            f"Red coverage={red:.2f}% (>=80%)."
        )

    return measurement


def assess_apple_ripeness(
    original_crop_bgr: np.ndarray,
    fruit_mask: Optional[np.ndarray] = None,
) -> Tuple[
    AppleRipenessResult,
    Dict[str, np.ndarray],
]:
    """
    YOLO mask
        -> Hue colour measurement
        -> red coverage
        -> Fuji maturity Index 1-5
    """
    result, diagnostics = (
        measure_apple_features(
            original_crop_bgr,
            fruit_mask=fruit_mask,
        )
    )

    result = classify_apple(
        result
    )

    return result, diagnostics


def save_diagnostics(
    out_dir: str,
    diagnostics: Dict[str, np.ndarray],
) -> None:
    """Save apple ripeness debug images."""
    import os

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    for name in (
        "fruit_mask",
        "hue_channel",
        "red_mask",
        "yellow_mask",
        "other_mask",
    ):
        if name in diagnostics:
            cv2.imwrite(
                os.path.join(
                    out_dir,
                    f"{name}.png",
                ),
                diagnostics[name],
            )

    if "bgr" in diagnostics:
        cv2.imwrite(
            os.path.join(
                out_dir,
                "input_original_colour.png",
            ),
            diagnostics["bgr"],
        )


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    SCRIPT_DIR = (
        Path(__file__).resolve().parent
    )

    DEFAULT_DEBUG_DIR = (
        SCRIPT_DIR
        / "apple_full_test_debug"
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Original-colour apple crop/image",
    )

    parser.add_argument(
        "--mask",
        help="Optional YOLO binary apple mask",
    )

    parser.add_argument(
        "--out",
        default=str(
            DEFAULT_DEBUG_DIR
        ),
        help=(
            "Default output: "
            "fruit_ripeness/apple_ripeness/apple_full_test_debug/"
        ),
    )

    args = parser.parse_args()

    img = cv2.imread(
        args.image
    )

    if img is None:
        raise SystemExit(
            f"Cannot read image: {args.image}"
        )

    mask = None

    if args.mask:
        mask = cv2.imread(
            args.mask,
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise SystemExit(
                f"Cannot read mask: {args.mask}"
            )

    result, diagnostics = (
        assess_apple_ripeness(
            img,
            fruit_mask=mask,
        )
    )

    print(
        "\nAPPLE RIPENESS RESULT"
    )
    print(
        "---------------------"
    )

    print(
        f"Fruit pixels    : "
        f"{result.total_fruit_pixels}"
    )

    print(
        f"Red pixels      : "
        f"{result.red_pixels}"
    )

    print(
        f"Yellow pixels   : "
        f"{result.yellow_pixels}"
    )

    print(
        f"Other pixels    : "
        f"{result.other_pixels}"
    )

    print(
        f"Red coverage    : "
        f"{result.red_pct:.2f}%"
    )

    print(
        f"Yellow coverage : "
        f"{result.yellow_pct:.2f}%"
    )

    print(
        f"Other coverage  : "
        f"{result.other_pct:.2f}%"
    )

    print(
        f"Prediction      : "
        f"{result.maturity_label}"
    )

    print(
        f"Rule basis      : "
        f"{result.rule_basis}"
    )

    save_diagnostics(
        args.out,
        diagnostics,
    )

    print(
        "\nDiagnostic masks saved to:",
        os.path.abspath(
            args.out
        ),
    )
