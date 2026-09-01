# pip install opencv-python numpy

"""
BANANA RIPENESS ANALYSIS - YELLOW + BROWN METHOD
=================================================

MAIN REFERENCE - BROWN SPOTTING / RIPENING
-------------------------------------------
Mendoza, F. & Aguilera, J. M. (2004),
"Application of Image Analysis for Classification of Ripening Bananas",
Journal of Food Science, 69(9), E471-E477.
DOI: 10.1111/j.1365-2621.2004.tb09932.x

The paper supports using banana peel colour together with brown-spot area as
ripening features. It defines %BSA as brown spots as a percentage of the total
banana area and reports that %BSA increases during late ripening, with final
values in the experiment ranging approximately from 20% to 50%.

The standard 7-stage visual scale discussed in the paper is:
    Index 1 : green
    Index 2 : green, traces of yellow
    Index 3 : more green than yellow
    Index 4 : more yellow than green
    Index 5 : green tip and yellow
    Index 6 : all yellow
    Index 7 : yellow, flecked with brown

IMPORTANT PROJECT INTERPRETATION
--------------------------------
The paper supports brown-area percentage as a useful ripeness feature and
reports late-ripening %BSA values reaching about 20-50%. It does NOT state an
official rule that exactly 20% brown must equal Index 7.

For this project, 20% is therefore used as an operational heavy-brown cutoff:

    FIRST: brown coverage >= 20% -> Index 7 - Overripe

This prevents a heavily blemished/brown banana with very little yellow left
from being incorrectly classified as Index 1 simply because yellow coverage is
low.

YELLOW MASK
-----------
The existing project yellow mask is kept unchanged:
    OpenCV H = 20-35
    OpenCV S = 40-255
    OpenCV V = 80-255

BROWN MASK
----------
The existing project brown mask is kept unchanged:
    OpenCV H = 0-25
    OpenCV S = 64-255
    OpenCV V = 0-153

Brown is detected before yellow, and brown pixels are removed from the yellow
mask so that the same pixel is not counted as both yellow and brown.

FINAL DECISION RULE ORDER
-------------------------
1. Brown coverage >= 20%
       -> Index 7 - Overripe / heavy brown

2. Small brown spotting on an otherwise developed yellow banana
       developed peel >= 95% AND brown > 1%
       -> Index 7 - Overripe / brown-spotted

3. Otherwise classify by yellow coverage:
       Yellow <= 1%          -> Index 1
       Yellow < 30%          -> Index 2
       30% <= Yellow < 50%   -> Index 3
       50% <= Yellow <= 80%  -> Index 4
       80% < Yellow < 95%    -> Index 5
       Yellow >= 95%         -> Index 6

PROJECT TOLERANCES
------------------
- Yellow <= 1% is treated as Index 1 to ignore isolated yellow noise pixels.
- Yellow >= 95% is treated as completely yellow because exact 100% is
  unrealistic in normal photographs.
- Brown > 1% is used for small brown spotting to avoid isolated brown noise.
- developed peel = yellow pixels + brown pixels.

PIPELINE
--------
Teammate YOLO banana mask
    -> original-colour banana pixels
    -> OpenCV HSV
    -> brown mask FIRST
    -> yellow mask with brown pixels excluded
    -> yellow/brown/developed-peel coverage
    -> Index 1-7

No CLAHE, white balance, or colour enhancement is applied to ripeness pixels.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Yellow mask
# ---------------------------------------------------------------------------

YELLOW_LOWER = np.array([20, 40, 80], dtype=np.uint8)
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Brown mask
# ---------------------------------------------------------------------------

BROWN_LOWER = np.array([0, 64, 0], dtype=np.uint8)
BROWN_UPPER = np.array([25, 255, 153], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Project operational thresholds
# ---------------------------------------------------------------------------

INDEX1_YELLOW_NOISE_MAX = 1.0
COMPLETE_YELLOW_MIN = 95.0
BROWN_SPOT_MIN = 1.0
HEAVY_BROWN_OVERRIPE_MIN = 20.0
DEVELOPED_PEEL_MIN = 95.0


@dataclass
class BananaRipenessResult:
    ripeness_index: int
    ripeness_label: str
    rule_basis: str

    total_fruit_pixels: int

    yellow_pixels: int
    brown_pixels: int
    other_pixels: int
    developed_peel_pixels: int

    yellow_pct: float
    brown_pct: float
    other_pct: float
    developed_peel_pct: float

    ripeness_factor: float


def _normalise_binary_mask(
    mask: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Convert a supplied YOLO mask to uint8 values 0 or 255."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape[:2] != target_shape:
        mask = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return (mask > 0).astype(np.uint8) * 255


def fruit_mask_from_black_background(
    masked_crop_bgr: np.ndarray,
) -> np.ndarray:
    """
    Fallback only.

    Final integration should pass the teammate YOLO mask directly.
    """
    if masked_crop_bgr is None or masked_crop_bgr.size == 0:
        raise ValueError("Input banana crop is empty")

    non_black = np.any(masked_crop_bgr != 0, axis=2)
    return non_black.astype(np.uint8) * 255


def prepare_banana_for_colour_analysis(
    original_crop_bgr: np.ndarray,
    fruit_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Prepare original-colour banana pixels for HSV analysis."""
    if original_crop_bgr is None or original_crop_bgr.size == 0:
        raise ValueError("Input banana crop is empty")

    h, w = original_crop_bgr.shape[:2]

    if fruit_mask is None:
        fruit_mask = fruit_mask_from_black_background(original_crop_bgr)
    else:
        fruit_mask = _normalise_binary_mask(
            fruit_mask,
            (h, w),
        )

    bgr = original_crop_bgr.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    return {
        "bgr": bgr,
        "hsv": hsv,
        "fruit_mask": fruit_mask,
    }


def _masked_range(
    hsv: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fruit_mask: np.ndarray,
) -> np.ndarray:
    """Create an HSV range mask restricted to the YOLO banana ROI."""
    raw = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(raw, fruit_mask)


def measure_banana_features(
    original_crop_bgr: np.ndarray,
    fruit_mask: Optional[np.ndarray] = None,
):
    """Measure yellow and brown coverage inside the YOLO banana mask."""
    data = prepare_banana_for_colour_analysis(
        original_crop_bgr,
        fruit_mask=fruit_mask,
    )

    hsv = data["hsv"]
    roi = data["fruit_mask"]

    total = int(cv2.countNonZero(roi))

    if total == 0:
        raise ValueError("YOLO banana mask contains no fruit pixels")

    # IMPORTANT: brown is calculated first.
    brown_mask = _masked_range(
        hsv,
        BROWN_LOWER,
        BROWN_UPPER,
        roi,
    )

    yellow_raw = _masked_range(
        hsv,
        YELLOW_LOWER,
        YELLOW_UPPER,
        roi,
    )

    # Brown gets priority so one pixel cannot be both yellow and brown.
    yellow_mask = cv2.bitwise_and(
        yellow_raw,
        cv2.bitwise_not(brown_mask),
    )
    yellow_mask = cv2.bitwise_and(yellow_mask, roi)

    # Developed peel = yellow or brown.
    developed_peel_mask = cv2.bitwise_or(
        yellow_mask,
        brown_mask,
    )
    developed_peel_mask = cv2.bitwise_and(
        developed_peel_mask,
        roi,
    )

    # Anything inside the fruit ROI that is neither yellow nor brown.
    other_mask = cv2.bitwise_and(
        roi,
        cv2.bitwise_not(developed_peel_mask),
    )

    yellow_pixels = int(cv2.countNonZero(yellow_mask))
    brown_pixels = int(cv2.countNonZero(brown_mask))
    developed_peel_pixels = int(cv2.countNonZero(developed_peel_mask))
    other_pixels = int(cv2.countNonZero(other_mask))

    yellow_pct = 100.0 * yellow_pixels / total
    brown_pct = 100.0 * brown_pixels / total
    developed_peel_pct = 100.0 * developed_peel_pixels / total
    other_pct = 100.0 * other_pixels / total

    # Brown spot area / total banana area.
    ripeness_factor = brown_pixels / total

    result = BananaRipenessResult(
        ripeness_index=0,
        ripeness_label="Not classified",
        rule_basis="",
        total_fruit_pixels=total,
        yellow_pixels=yellow_pixels,
        brown_pixels=brown_pixels,
        other_pixels=other_pixels,
        developed_peel_pixels=developed_peel_pixels,
        yellow_pct=yellow_pct,
        brown_pct=brown_pct,
        other_pct=other_pct,
        developed_peel_pct=developed_peel_pct,
        ripeness_factor=ripeness_factor,
    )

    diagnostics = {
        "bgr": data["bgr"],
        "fruit_mask": roi,
        "yellow_mask": yellow_mask,
        "brown_mask": brown_mask,
        "developed_peel_mask": developed_peel_mask,
        "other_mask": other_mask,
    }

    return result, diagnostics


def classify_banana(
    measurement: BananaRipenessResult,
) -> BananaRipenessResult:
    """
    Apply banana Index 1-7 with brown checked FIRST.

    Exact rule order:
        Brown >= 20%             -> Index 7
        Small brown on developed yellow peel -> Index 7
        Yellow <= 1%             -> Index 1
        Yellow < 30%             -> Index 2
        30% <= Yellow < 50%      -> Index 3
        50% <= Yellow <= 80%     -> Index 4
        80% < Yellow < 95%       -> Index 5
        Yellow >= 95%            -> Index 6
    """
    yellow = measurement.yellow_pct
    brown = measurement.brown_pct
    developed = measurement.developed_peel_pct

    # ------------------------------------------------------------------
    # FIRST CHECK: heavy brown area overrides yellow percentage.
    # This solves the case where a very blemished banana has almost no
    # remaining yellow and would otherwise incorrectly fall into Index 1.
    # ------------------------------------------------------------------
    if brown >= HEAVY_BROWN_OVERRIPE_MIN:
        measurement.ripeness_index = 7
        measurement.ripeness_label = "Index 7 - Overripe / heavy brown"
        measurement.rule_basis = (
            f"Brown coverage={brown:.2f}% >= "
            f"{HEAVY_BROWN_OVERRIPE_MIN:.1f}% brown-first threshold."
        )

    # Small brown spotting on an otherwise developed yellow banana.
    elif developed >= DEVELOPED_PEEL_MIN and brown > BROWN_SPOT_MIN:
        measurement.ripeness_index = 7
        measurement.ripeness_label = "Index 7 - Overripe / brown-spotted"
        measurement.rule_basis = (
            f"Developed peel={developed:.2f}% >= "
            f"{DEVELOPED_PEEL_MIN:.1f}% and brown={brown:.2f}% > "
            f"{BROWN_SPOT_MIN:.1f}% small-spot tolerance."
        )

    # Otherwise classify using yellow coverage.
    elif yellow >= COMPLETE_YELLOW_MIN:
        measurement.ripeness_index = 6
        measurement.ripeness_label = "Index 6 - Completely yellow"
        measurement.rule_basis = (
            f"Yellow coverage={yellow:.2f}% >= "
            f"{COMPLETE_YELLOW_MIN:.1f}%."
        )

    elif yellow > 80.0:
        measurement.ripeness_index = 5
        measurement.ripeness_label = "Index 5 - More than 80% yellow"
        measurement.rule_basis = (
            f"Yellow coverage={yellow:.2f}% is >80% and <"
            f"{COMPLETE_YELLOW_MIN:.1f}%."
        )

    elif yellow >= 50.0:
        measurement.ripeness_index = 4
        measurement.ripeness_label = "Index 4 - 50-80% yellow"
        measurement.rule_basis = (
            f"Yellow coverage={yellow:.2f}% is 50-80%."
        )

    elif yellow >= 30.0:
        measurement.ripeness_index = 3
        measurement.ripeness_label = "Index 3 - 30-50% yellow"
        measurement.rule_basis = (
            f"Yellow coverage={yellow:.2f}% is >=30% and <50%."
        )

    elif yellow > INDEX1_YELLOW_NOISE_MAX:
        measurement.ripeness_index = 2
        measurement.ripeness_label = "Index 2 - Less than 30% yellow"
        measurement.rule_basis = (
            f"Yellow coverage={yellow:.2f}% is >"
            f"{INDEX1_YELLOW_NOISE_MAX:.1f}% and <30%."
        )

    else:
        measurement.ripeness_index = 1
        measurement.ripeness_label = "Index 1 - Completely green"
        measurement.rule_basis = (
            f"Brown coverage={brown:.2f}% < "
            f"{HEAVY_BROWN_OVERRIPE_MIN:.1f}% and yellow coverage="
            f"{yellow:.2f}% <= {INDEX1_YELLOW_NOISE_MAX:.1f}%."
        )

    return measurement


def assess_banana_ripeness(
    original_crop_bgr: np.ndarray,
    fruit_mask: Optional[np.ndarray] = None,
):
    """YOLO mask -> HSV yellow/brown masks -> Index 1-7."""
    result, diagnostics = measure_banana_features(
        original_crop_bgr,
        fruit_mask=fruit_mask,
    )

    result = classify_banana(result)

    return result, diagnostics


def save_diagnostics(
    out_dir: str,
    diagnostics: Dict[str, np.ndarray],
) -> None:
    """Save banana ripeness debug images."""
    import os

    os.makedirs(out_dir, exist_ok=True)

    for name in (
        "fruit_mask",
        "yellow_mask",
        "brown_mask",
        "developed_peel_mask",
        "other_mask",
    ):
        cv2.imwrite(
            os.path.join(out_dir, f"{name}.png"),
            diagnostics[name],
        )

    cv2.imwrite(
        os.path.join(out_dir, "input_original_colour.png"),
        diagnostics["bgr"],
    )


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_DEBUG_DIR = SCRIPT_DIR / "banana_full_test_debug"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Original-colour banana crop/image",
    )

    parser.add_argument(
        "--mask",
        help="Optional YOLO binary banana mask",
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_DEBUG_DIR),
    )

    args = parser.parse_args()

    img = cv2.imread(args.image)

    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    mask = None

    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise SystemExit(f"Cannot read mask: {args.mask}")

    result, diagnostics = assess_banana_ripeness(
        img,
        fruit_mask=mask,
    )

    print("\nBANANA RIPENESS RESULT")
    print("----------------------")
    print(f"Fruit pixels           : {result.total_fruit_pixels}")
    print(f"Yellow pixels          : {result.yellow_pixels}")
    print(f"Brown pixels           : {result.brown_pixels}")
    print(f"Developed peel pixels  : {result.developed_peel_pixels}")
    print(f"Other pixels           : {result.other_pixels}")
    print(f"Yellow coverage        : {result.yellow_pct:.2f}%")
    print(f"Brown coverage         : {result.brown_pct:.2f}%")
    print(f"Developed peel coverage: {result.developed_peel_pct:.2f}%")
    print(f"Other coverage         : {result.other_pct:.2f}%")
    print(f"Ripeness Factor (RF)   : {result.ripeness_factor:.6f}")
    print(f"Prediction             : {result.ripeness_label}")
    print(f"Rule basis             : {result.rule_basis}")

    save_diagnostics(args.out, diagnostics)

    print(
        "\nDiagnostic masks saved to:",
        os.path.abspath(args.out),
    )
