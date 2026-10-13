from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

YELLOW_LOWER = np.array([15, 135, 80], dtype=np.uint8)
YELLOW_UPPER = np.array([30, 255, 255], dtype=np.uint8)


STAGE1_MAX = 10.0
STAGE2_MAX = 40.0
STAGE3_MAX = 70.0


@dataclass
class MangoRipenessResult:
    maturity_stage: int
    maturity_label: str
    rule_basis: str

    total_fruit_pixels: int

    yellow_pixels: int
    other_pixels: int

    yellow_pct: float
    other_pct: float


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
        raise ValueError("Input mango crop is empty")

    non_black = np.any(masked_crop_bgr != 0, axis=2)

    return non_black.astype(np.uint8) * 255


def prepare_mango_for_colour_analysis(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    if original_crop_bgr is None or original_crop_bgr.size == 0:
        raise ValueError("Input mango crop is empty")

    h, w = original_crop_bgr.shape[:2]

    if fruit_mask is None:
        fruit_mask = fruit_mask_from_black_background(original_crop_bgr)
    else:
        fruit_mask = _normalise_binary_mask(fruit_mask, (h, w))

    bgr = original_crop_bgr.copy()

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    return {"bgr": bgr, "hsv": hsv, "fruit_mask": fruit_mask}


def measure_mango_features(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
):
    data = prepare_mango_for_colour_analysis(original_crop_bgr, fruit_mask=fruit_mask)

    hsv = data["hsv"]
    roi_mask = data["fruit_mask"]

    total = int(cv2.countNonZero(roi_mask))

    if total == 0:
        raise ValueError("YOLO mango mask contains no fruit pixels")

    yellow_raw = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    yellow_mask = cv2.bitwise_and(yellow_raw, roi_mask)
    other_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(yellow_mask))

    yellow_pixels = int(cv2.countNonZero(yellow_mask))
    other_pixels = int(cv2.countNonZero(other_mask))

    yellow_pct = 100.0 * yellow_pixels / total
    other_pct = 100.0 * other_pixels / total

    result = MangoRipenessResult(
        maturity_stage=0,
        maturity_label="Not classified",
        rule_basis="",
        total_fruit_pixels=total,
        yellow_pixels=yellow_pixels,
        other_pixels=other_pixels,
        yellow_pct=yellow_pct,
        other_pct=other_pct,
    )

    diagnostics = {
        "bgr": data["bgr"],
        "fruit_mask": roi_mask,
        "yellow_mask": yellow_mask,
        "other_mask": other_mask,
    }

    return result, diagnostics


def classify_mango(measurement: MangoRipenessResult) -> MangoRipenessResult:
    yellow = measurement.yellow_pct

    if yellow <= STAGE1_MAX:
        measurement.maturity_stage = 1
        measurement.maturity_label = "Stage 1"
        measurement.rule_basis = f"Yellow coverage={yellow:.2f}% (0-10%)."

    elif yellow <= STAGE2_MAX:
        measurement.maturity_stage = 2
        measurement.maturity_label = "Stage 2"
        measurement.rule_basis = f"Yellow coverage={yellow:.2f}% (>10% and <=40%)."

    elif yellow <= STAGE3_MAX:
        measurement.maturity_stage = 3
        measurement.maturity_label = "Stage 3"
        measurement.rule_basis = f"Yellow coverage={yellow:.2f}% (>40% and <=70%)."

    else:
        measurement.maturity_stage = 4
        measurement.maturity_label = "Stage 4"
        measurement.rule_basis = f"Yellow coverage={yellow:.2f}% (>70%)."

    return measurement


def assess_mango_ripeness(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
):
    result, diagnostics = measure_mango_features(
        original_crop_bgr, fruit_mask=fruit_mask
    )

    result = classify_mango(result)

    return result, diagnostics


def save_diagnostics(out_dir: str, diagnostics: Dict[str, np.ndarray]) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)

    for name in ("fruit_mask", "yellow_mask", "other_mask"):
        if name in diagnostics:
            cv2.imwrite(os.path.join(out_dir, f"{name}.png"), diagnostics[name])

    if "bgr" in diagnostics:
        cv2.imwrite(
            os.path.join(out_dir, "input_original_colour.png"), diagnostics["bgr"]
        )


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_DEBUG_DIR = SCRIPT_DIR / "mango_full_test_debug"

    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="Original-colour mango crop/image")
    parser.add_argument("--mask", help="Optional YOLO binary mango mask")

    parser.add_argument(
        "--out",
        default=str(DEFAULT_DEBUG_DIR),
        help="Default: fruit_ripeness/mango_ripeness/mango_full_test_debug/",
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

    result, diagnostics = assess_mango_ripeness(img, fruit_mask=mask)

    print("\nMANGO RIPENESS RESULT")
    print("----------------------")
    print(f"Fruit pixels    : {result.total_fruit_pixels}")
    print(
        f"Yellow pixels   : "
        f"{result.yellow_pixels} "
        "(H 15-30, S 135-255, V 80-255)"
    )
    print(f"Other pixels    : {result.other_pixels}")
    print(f"Yellow coverage : {result.yellow_pct:.2f}%")
    print(f"Other coverage  : {result.other_pct:.2f}%")
    print(f"Prediction      : {result.maturity_label}")
    print(f"Rule basis      : {result.rule_basis}")

    save_diagnostics(args.out, diagnostics)

    print("\nDiagnostic masks saved to:", os.path.abspath(args.out))
