from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

YELLOW_LOWER = np.array([20, 40, 80], dtype=np.uint8)
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)


BROWN_LOWER = np.array([0, 64, 0], dtype=np.uint8)
BROWN_UPPER = np.array([25, 255, 153], dtype=np.uint8)


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
    mask: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape[:2] != target_shape:
        mask = cv2.resize(
            mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
        )

    return (mask > 0).astype(np.uint8) * 255


def fruit_mask_from_black_background(masked_crop_bgr: np.ndarray) -> np.ndarray:
    if masked_crop_bgr is None or masked_crop_bgr.size == 0:
        raise ValueError("Input banana crop is empty")

    non_black = np.any(masked_crop_bgr != 0, axis=2)
    return non_black.astype(np.uint8) * 255


def prepare_banana_for_colour_analysis(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    if original_crop_bgr is None or original_crop_bgr.size == 0:
        raise ValueError("Input banana crop is empty")

    h, w = original_crop_bgr.shape[:2]

    if fruit_mask is None:
        fruit_mask = fruit_mask_from_black_background(original_crop_bgr)
    else:
        fruit_mask = _normalise_binary_mask(fruit_mask, (h, w))

    bgr = original_crop_bgr.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    return {"bgr": bgr, "hsv": hsv, "fruit_mask": fruit_mask}


def _masked_range(
    hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray, fruit_mask: np.ndarray
) -> np.ndarray:
    raw = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(raw, fruit_mask)


def measure_banana_features(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
):
    data = prepare_banana_for_colour_analysis(original_crop_bgr, fruit_mask=fruit_mask)

    hsv = data["hsv"]
    roi = data["fruit_mask"]

    total = int(cv2.countNonZero(roi))

    if total == 0:
        raise ValueError("YOLO banana mask contains no fruit pixels")

    brown_mask = _masked_range(hsv, BROWN_LOWER, BROWN_UPPER, roi)
    yellow_raw = _masked_range(hsv, YELLOW_LOWER, YELLOW_UPPER, roi)

    yellow_mask = cv2.bitwise_and(yellow_raw, cv2.bitwise_not(brown_mask))
    yellow_mask = cv2.bitwise_and(yellow_mask, roi)

    developed_peel_mask = cv2.bitwise_or(yellow_mask, brown_mask)
    developed_peel_mask = cv2.bitwise_and(developed_peel_mask, roi)
    other_mask = cv2.bitwise_and(roi, cv2.bitwise_not(developed_peel_mask))

    yellow_pixels = int(cv2.countNonZero(yellow_mask))
    brown_pixels = int(cv2.countNonZero(brown_mask))
    developed_peel_pixels = int(cv2.countNonZero(developed_peel_mask))
    other_pixels = int(cv2.countNonZero(other_mask))

    yellow_pct = 100.0 * yellow_pixels / total
    brown_pct = 100.0 * brown_pixels / total
    developed_peel_pct = 100.0 * developed_peel_pixels / total
    other_pct = 100.0 * other_pixels / total
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


def classify_banana(measurement: BananaRipenessResult) -> BananaRipenessResult:
    yellow = measurement.yellow_pct
    brown = measurement.brown_pct
    developed = measurement.developed_peel_pct

    if brown >= HEAVY_BROWN_OVERRIPE_MIN:
        measurement.ripeness_index = 7
        measurement.ripeness_label = "Index 7 - Overripe / heavy brown"
        measurement.rule_basis = (
            f"Brown coverage={brown:.2f}% >= "
            f"{HEAVY_BROWN_OVERRIPE_MIN:.1f}% brown-first threshold."
        )

    elif developed >= DEVELOPED_PEEL_MIN and brown > BROWN_SPOT_MIN:
        measurement.ripeness_index = 7
        measurement.ripeness_label = "Index 7 - Overripe / brown-spotted"
        measurement.rule_basis = (
            f"Developed peel={developed:.2f}% >= "
            f"{DEVELOPED_PEEL_MIN:.1f}% and brown={brown:.2f}% > "
            f"{BROWN_SPOT_MIN:.1f}% small-spot tolerance."
        )

    elif yellow >= COMPLETE_YELLOW_MIN:
        measurement.ripeness_index = 6
        measurement.ripeness_label = "Index 6 - Completely yellow"
        measurement.rule_basis = (
            f"Yellow coverage={yellow:.2f}% >= {COMPLETE_YELLOW_MIN:.1f}%."
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
        measurement.rule_basis = f"Yellow coverage={yellow:.2f}% is 50-80%."

    elif yellow >= 30.0:
        measurement.ripeness_index = 3
        measurement.ripeness_label = "Index 3 - 30-50% yellow"
        measurement.rule_basis = f"Yellow coverage={yellow:.2f}% is >=30% and <50%."

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
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
):
    result, diagnostics = measure_banana_features(
        original_crop_bgr, fruit_mask=fruit_mask
    )

    result = classify_banana(result)

    return result, diagnostics


def save_diagnostics(out_dir: str, diagnostics: Dict[str, np.ndarray]) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)

    for name in (
        "fruit_mask",
        "yellow_mask",
        "brown_mask",
        "developed_peel_mask",
        "other_mask",
    ):
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), diagnostics[name])

    cv2.imwrite(os.path.join(out_dir, "input_original_colour.png"), diagnostics["bgr"])


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_DEBUG_DIR = SCRIPT_DIR / "banana_full_test_debug"

    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="Original-colour banana crop/image")
    parser.add_argument("--mask", help="Optional YOLO binary banana mask")
    parser.add_argument("--out", default=str(DEFAULT_DEBUG_DIR))

    args = parser.parse_args()

    img = cv2.imread(args.image)

    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    mask = None

    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise SystemExit(f"Cannot read mask: {args.mask}")

    result, diagnostics = assess_banana_ripeness(img, fruit_mask=mask)

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

    print("\nDiagnostic masks saved to:", os.path.abspath(args.out))
