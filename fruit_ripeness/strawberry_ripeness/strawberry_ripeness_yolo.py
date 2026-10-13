from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

RED_RANGES = (((0, 64, 0), (14, 255, 255)), ((167, 64, 0), (179, 255, 255)))

DARK_RED_RANGES = (((0, 64, 0), (10, 255, 90)), ((167, 64, 0), (179, 255, 90)))

FAMA_RED_BOUNDARY = 50.0

FUKUDA_DARK_RED_OVERRIPE = 90.0

SUPPLEMENTARY_UNRIPE_RED_MAX = 10.0


@dataclass
class StrawberryRipenessResult:
    maturity_index: Optional[int]
    maturity_label: str
    rule_basis: str

    total_flesh_pixels: int

    red_pixels: int
    dark_red_pixels: int
    red_pct: float
    dark_red_pct: float
    normal_red_pct: float
    other_pct: float
    boundary_case: bool = False


def _combine_hsv_ranges(hsv: np.ndarray, ranges) -> np.ndarray:
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        part = cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        )
        combined = cv2.bitwise_or(combined, part)

    return combined


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
        raise ValueError("Input strawberry crop is empty")

    non_black = np.any(masked_crop_bgr != 0, axis=2)
    return non_black.astype(np.uint8) * 255


def prepare_strawberry_for_colour_analysis(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    if original_crop_bgr is None or original_crop_bgr.size == 0:
        raise ValueError("Input strawberry crop is empty")

    h, w = original_crop_bgr.shape[:2]

    if fruit_mask is None:
        fruit_mask = fruit_mask_from_black_background(original_crop_bgr)
    else:
        fruit_mask = _normalise_binary_mask(fruit_mask, (h, w))

    original_colour = original_crop_bgr.copy()
    hsv = cv2.cvtColor(original_colour, cv2.COLOR_BGR2HSV)

    return {"bgr": original_colour, "hsv": hsv, "fruit_mask": fruit_mask}


def measure_strawberry_colour_percentages(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
):
    data = prepare_strawberry_for_colour_analysis(
        original_crop_bgr, fruit_mask=fruit_mask
    )

    hsv = data["hsv"]
    roi_mask = data["fruit_mask"]

    total = int(cv2.countNonZero(roi_mask))

    if total == 0:
        raise ValueError("YOLO fruit mask contains no strawberry pixels")

    red_mask = _combine_hsv_ranges(hsv, RED_RANGES)
    dark_red_mask = _combine_hsv_ranges(hsv, DARK_RED_RANGES)

    red_mask = cv2.bitwise_and(red_mask, roi_mask)
    dark_red_mask = cv2.bitwise_and(dark_red_mask, roi_mask)

    red_pixels = int(cv2.countNonZero(red_mask))
    dark_red_pixels = int(cv2.countNonZero(dark_red_mask))

    red_pct = 100.0 * red_pixels / total
    dark_red_pct = 100.0 * dark_red_pixels / total

    normal_red_pct = max(0.0, red_pct - dark_red_pct)
    other_pct = max(0.0, 100.0 - red_pct)

    result = StrawberryRipenessResult(
        maturity_index=None,
        maturity_label="Not classified",
        rule_basis="",
        total_flesh_pixels=total,
        red_pixels=red_pixels,
        dark_red_pixels=dark_red_pixels,
        red_pct=red_pct,
        dark_red_pct=dark_red_pct,
        normal_red_pct=normal_red_pct,
        other_pct=other_pct,
    )

    data["red_mask"] = red_mask
    data["dark_red_mask"] = dark_red_mask

    return result, data


def classify_strawberry(
    measurement: StrawberryRipenessResult, use_supplementary_index1_rule: bool = False
) -> StrawberryRipenessResult:
    r = measurement.red_pct
    dr = measurement.dark_red_pct

    if dr >= FUKUDA_DARK_RED_OVERRIPE:
        measurement.maturity_index = 4
        measurement.maturity_label = "Index 4 - Overmature (visual estimate)"
        measurement.rule_basis = (
            "FAMA Index 4 requires the whole surface to be dark red and juicy. "
            "RGB can assess colour but not juiciness. >=90% dark-red is the "
            "operational colour rule from Fukuda et al. (2025), not FAMA."
        )
        return measurement

    if np.isclose(r, FAMA_RED_BOUNDARY, atol=1e-9):
        measurement.maturity_index = None
        measurement.maturity_label = "Boundary case - exactly 50% red"
        measurement.rule_basis = (
            "FAMA states <50% red for Index 2 and >50% red for Index 3; "
            "exactly 50% is not specified."
        )
        measurement.boundary_case = True
        return measurement

    if r > FAMA_RED_BOUNDARY:
        measurement.maturity_index = 3
        measurement.maturity_label = "Index 3 - Mature"
        measurement.rule_basis = "FAMA FS069:2013: visible surface >50% red."
        return measurement

    if use_supplementary_index1_rule and r < SUPPLEMENTARY_UNRIPE_RED_MAX:
        measurement.maturity_index = 1
        measurement.maturity_label = "Index 1 - Unripe (supplementary operational rule)"
        measurement.rule_basis = (
            "FAMA describes Index 1 as greenish-cream. "
            "Because FAMA gives no numeric HSV/red-percentage boundary, "
            "<10% red is used only as a supplementary operational rule."
        )
        return measurement

    if use_supplementary_index1_rule:
        measurement.maturity_index = 2
        measurement.maturity_label = "Index 2 - Mature"
        measurement.rule_basis = (
            "FAMA FS069:2013: <50% red; the supplementary <10% rule "
            "has already separated Index-1-like cases."
        )
        return measurement

    measurement.maturity_index = None
    measurement.maturity_label = "Index 1 / Index 2 unresolved in strict mode"
    measurement.rule_basis = (
        "FAMA distinguishes Index 1 by greenish-cream surface and Index 2 "
        "by <50% red, but the current references do not provide a validated "
        "pixel-level HSV range for greenish-cream strawberry flesh."
    )

    return measurement


def assess_strawberry_ripeness(
    original_crop_bgr: np.ndarray,
    fruit_mask: Optional[np.ndarray] = None,
    use_supplementary_index1_rule: bool = False,
):
    result, diagnostics = measure_strawberry_colour_percentages(
        original_crop_bgr, fruit_mask=fruit_mask
    )

    result = classify_strawberry(
        result, use_supplementary_index1_rule=use_supplementary_index1_rule
    )

    return result, diagnostics


def save_diagnostics(out_dir: str, diagnostics: Dict[str, np.ndarray]) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)

    for name in ("fruit_mask", "red_mask", "dark_red_mask"):
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
    DEFAULT_DEBUG_DIR = SCRIPT_DIR / "strawberry_full_test_debug"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help=(
            "Original-colour strawberry crop. "
            "Pass --mask with the YOLO mask for best results."
        ),
    )

    parser.add_argument("--mask", help="Binary YOLO strawberry mask")

    parser.add_argument(
        "--out",
        default=str(DEFAULT_DEBUG_DIR),
        help=(
            "Diagnostic output folder. By default, results are stored in "
            "fruit_ripeness/strawberry_ripeness/strawberry_full_test_debug/."
        ),
    )

    parser.add_argument(
        "--supplementary-index1",
        action="store_true",
        help="Use the non-FAMA <10%% red supplementary rule for Index 1",
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

    result, diagnostics = assess_strawberry_ripeness(
        img, fruit_mask=mask, use_supplementary_index1_rule=args.supplementary_index1
    )

    print("\nSTRAWBERRY RIPENESS RESULT")
    print("---------------------------")
    print(f"YOLO ROI pixels : {result.total_flesh_pixels}")
    print(f"Red pixels      : {result.red_pixels}")
    print(f"Dark-red pixels : {result.dark_red_pixels}")
    print(f"Red coverage    : {result.red_pct:.2f}%")
    print(f"Normal red      : {result.normal_red_pct:.2f}%")
    print(f"Dark red        : {result.dark_red_pct:.2f}%")
    print(f"Other           : {result.other_pct:.2f}%")
    print(f"Prediction      : {result.maturity_label}")
    print(f"Rule basis      : {result.rule_basis}")

    save_diagnostics(args.out, diagnostics)

    print("\nDiagnostic masks saved to:", os.path.abspath(args.out))
