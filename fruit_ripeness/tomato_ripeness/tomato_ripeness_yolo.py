from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

GREEN_RANGE = ((35, 40, 40), (85, 255, 255))

TURNING_RANGE = ((20, 40, 40), (35, 255, 255))

PINK_RANGE = ((10, 40, 40), (20, 255, 255))

RED_RANGES = (((0, 40, 40), (10, 255, 255)), ((170, 40, 40), (179, 255, 255)))

EARLY_GREEN_COLOUR_BREAK_MAX = 1.0

INDEX3_BOUNDARY = 50.0
INDEX4_BOUNDARY = 75.0
FULL_RED_BOUNDARY = 90.0

BRIGHT_RED_LSTAR = 41.8
DARK_RED_LSTAR = 39.6
DARK_LSTAR_THRESHOLD = (BRIGHT_RED_LSTAR + DARK_RED_LSTAR) / 2.0


@dataclass
class TomatoRipenessResult:
    maturity_index: Optional[int]
    maturity_label: str
    rule_basis: str

    total_fruit_pixels: int

    green_pixels: int
    turning_pixels: int
    pink_pixels: int
    mature_red_pixels: int
    colour_break_pixels: int
    red_orange_pixels: int
    other_pixels: int

    green_pct: float
    turning_pct: float
    pink_pct: float
    mature_red_pct: float
    colour_break_pct: float
    red_orange_pct: float
    other_pct: float

    mean_lstar_red_orange: Optional[float]
    median_lstar_red_orange: Optional[float]

    boundary_case: bool = False


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
        raise ValueError("Input tomato crop is empty")

    non_black = np.any(masked_crop_bgr != 0, axis=2)
    return non_black.astype(np.uint8) * 255


def _single_hsv_range_mask(hsv: np.ndarray, hsv_range) -> np.ndarray:
    lower, upper = hsv_range
    return cv2.inRange(
        hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
    )


def _combine_hsv_ranges(hsv: np.ndarray, ranges) -> np.ndarray:
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        part = cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        )
        combined = cv2.bitwise_or(combined, part)

    return combined


def prepare_tomato_for_colour_analysis(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    if original_crop_bgr is None or original_crop_bgr.size == 0:
        raise ValueError("Input tomato crop is empty")

    h, w = original_crop_bgr.shape[:2]

    if fruit_mask is None:
        fruit_mask = fruit_mask_from_black_background(original_crop_bgr)
    else:
        fruit_mask = _normalise_binary_mask(fruit_mask, (h, w))

    bgr = original_crop_bgr.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    return {"bgr": bgr, "hsv": hsv, "lab": lab, "fruit_mask": fruit_mask}


def measure_tomato_features(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
) -> Tuple[TomatoRipenessResult, Dict[str, np.ndarray]]:
    data = prepare_tomato_for_colour_analysis(original_crop_bgr, fruit_mask=fruit_mask)

    hsv = data["hsv"]
    lab = data["lab"]
    roi_mask = data["fruit_mask"]

    total = int(cv2.countNonZero(roi_mask))
    if total == 0:
        raise ValueError("YOLO tomato mask contains no fruit pixels")

    green_raw = cv2.bitwise_and(_single_hsv_range_mask(hsv, GREEN_RANGE), roi_mask)
    turning_raw = cv2.bitwise_and(_single_hsv_range_mask(hsv, TURNING_RANGE), roi_mask)
    pink_raw = cv2.bitwise_and(_single_hsv_range_mask(hsv, PINK_RANGE), roi_mask)
    red_raw = cv2.bitwise_and(_combine_hsv_ranges(hsv, RED_RANGES), roi_mask)

    mature_red_mask = red_raw

    pink_mask = cv2.bitwise_and(pink_raw, cv2.bitwise_not(mature_red_mask))

    red_or_pink = cv2.bitwise_or(mature_red_mask, pink_mask)

    turning_mask = cv2.bitwise_and(turning_raw, cv2.bitwise_not(red_or_pink))

    occupied_before_green = cv2.bitwise_or(red_or_pink, turning_mask)

    green_mask = cv2.bitwise_and(green_raw, cv2.bitwise_not(occupied_before_green))

    colour_break_mask = cv2.bitwise_or(turning_mask, pink_mask)
    colour_break_mask = cv2.bitwise_or(colour_break_mask, mature_red_mask)
    red_orange_mask = cv2.bitwise_or(pink_mask, mature_red_mask)
    classified_mask = cv2.bitwise_or(green_mask, colour_break_mask)

    other_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(classified_mask))

    green_pixels = int(cv2.countNonZero(green_mask))
    turning_pixels = int(cv2.countNonZero(turning_mask))
    pink_pixels = int(cv2.countNonZero(pink_mask))
    mature_pixels = int(cv2.countNonZero(mature_red_mask))
    colour_break_pixels = int(cv2.countNonZero(colour_break_mask))
    red_orange_pixels = int(cv2.countNonZero(red_orange_mask))
    other_pixels = int(cv2.countNonZero(other_mask))

    green_pct = 100.0 * green_pixels / total
    turning_pct = 100.0 * turning_pixels / total
    pink_pct = 100.0 * pink_pixels / total
    mature_pct = 100.0 * mature_pixels / total
    colour_break_pct = 100.0 * colour_break_pixels / total
    red_orange_pct = 100.0 * red_orange_pixels / total
    other_pct = 100.0 * other_pixels / total

    mean_lstar = None
    median_lstar = None

    mature_red_bool = mature_red_mask > 0
    if np.any(mature_red_bool):
        l_opencv = lab[:, :, 0][mature_red_bool].astype(np.float32)
        lstar = l_opencv * (100.0 / 255.0)
        mean_lstar = float(np.mean(lstar))
        median_lstar = float(np.median(lstar))

    result = TomatoRipenessResult(
        maturity_index=None,
        maturity_label="Not classified",
        rule_basis="",
        total_fruit_pixels=total,
        green_pixels=green_pixels,
        turning_pixels=turning_pixels,
        pink_pixels=pink_pixels,
        mature_red_pixels=mature_pixels,
        colour_break_pixels=colour_break_pixels,
        red_orange_pixels=red_orange_pixels,
        other_pixels=other_pixels,
        green_pct=green_pct,
        turning_pct=turning_pct,
        pink_pct=pink_pct,
        mature_red_pct=mature_pct,
        colour_break_pct=colour_break_pct,
        red_orange_pct=red_orange_pct,
        other_pct=other_pct,
        mean_lstar_red_orange=mean_lstar,
        median_lstar_red_orange=median_lstar,
    )

    data.update(
        {
            "green_mask": green_mask,
            "turning_mask": turning_mask,
            "pink_mask": pink_mask,
            "mature_red_mask": mature_red_mask,
            "colour_break_mask": colour_break_mask,
            "red_orange_mask": red_orange_mask,
            "other_mask": other_mask,
            "lab_l_channel": lab[:, :, 0],
        }
    )

    return result, data


def classify_tomato(measurement: TomatoRipenessResult) -> TomatoRipenessResult:
    colour_break = measurement.colour_break_pct
    red_orange = measurement.red_orange_pct
    mature = measurement.mature_red_pct
    green = measurement.green_pct
    other = measurement.other_pct

    if mature > FULL_RED_BOUNDARY:
        if measurement.median_lstar_red_orange is None:
            measurement.maturity_index = None
            measurement.maturity_label = "Index 5 / Index 6 unresolved"
            measurement.rule_basis = (
                "Mature/red coverage is >90%, but no mature-red L* "
                "measurement is available."
            )
            return measurement

        if measurement.median_lstar_red_orange <= DARK_LSTAR_THRESHOLD:
            measurement.maturity_index = 6
            measurement.maturity_label = "Index 6 - Overmature (visual estimate)"
            measurement.rule_basis = (
                f"Mature/red coverage={mature:.2f}% (>90%). "
                f"Median mature-red L*="
                f"{measurement.median_lstar_red_orange:.2f} <= 40.7. "
                "40.7 is derived from bright-red L*=41.8 and "
                "dark-red L*=39.6."
            )
        else:
            measurement.maturity_index = 5
            measurement.maturity_label = "Index 5 - Fully ripe (visual estimate)"
            measurement.rule_basis = (
                f"Mature/red coverage={mature:.2f}% (>90%). "
                f"Median mature-red L*="
                f"{measurement.median_lstar_red_orange:.2f} > 40.7. "
                "40.7 is derived from bright-red L*=41.8 and "
                "dark-red L*=39.6."
            )

        return measurement

    if red_orange > INDEX4_BOUNDARY and mature <= FULL_RED_BOUNDARY:
        measurement.maturity_index = 4
        measurement.maturity_label = "Index 4 - Three-quarter ripe"
        measurement.rule_basis = (
            f"Red/orange coverage (pink + mature/red)="
            f"{red_orange:.2f}% (>75%). "
            f"Mature/red coverage={mature:.2f}% (<=90%)."
        )
        return measurement

    if colour_break >= INDEX3_BOUNDARY and colour_break <= INDEX4_BOUNDARY:
        measurement.maturity_index = None
        measurement.maturity_label = "Index 3 / Index 4 transition"
        measurement.rule_basis = (
            f"Colour-break coverage (turning + pink + mature/red)="
            f"{colour_break:.2f}% (50% to 75%)."
        )
        measurement.boundary_case = True
        return measurement

    if colour_break > EARLY_GREEN_COLOUR_BREAK_MAX and colour_break < INDEX3_BOUNDARY:
        measurement.maturity_index = 3
        measurement.maturity_label = "Index 3 - Colour break"
        measurement.rule_basis = (
            f"Colour-break coverage (turning + pink + mature/red)="
            f"{colour_break:.2f}% (>1% and <50%)."
        )
        return measurement

    if colour_break <= EARLY_GREEN_COLOUR_BREAK_MAX and green > other:
        measurement.maturity_index = None
        measurement.maturity_label = "Index 1 / Index 2 - Green stage (combined)"
        measurement.rule_basis = (
            f"Colour-break coverage={colour_break:.2f}% "
            f"(<=1% noise tolerance), green={green:.2f}% and "
            f"other={other:.2f}%. "
            "Index 1/2 remain combined because FAMA separates them "
            "by dull versus glossy surface appearance."
        )
        return measurement

    measurement.maturity_index = None
    measurement.maturity_label = "Unresolved by current tomato rules"
    measurement.rule_basis = (
        f"Green={measurement.green_pct:.2f}%, "
        f"turning={measurement.turning_pct:.2f}%, "
        f"pink={measurement.pink_pct:.2f}%, "
        f"mature/red={measurement.mature_red_pct:.2f}%, "
        f"colour-break={measurement.colour_break_pct:.2f}%, "
        f"red-orange={measurement.red_orange_pct:.2f}%, "
        f"other={measurement.other_pct:.2f}%. "
        "The image does not satisfy one complete project rule."
    )

    return measurement


def assess_tomato_ripeness(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
) -> Tuple[TomatoRipenessResult, Dict[str, np.ndarray]]:
    result, diagnostics = measure_tomato_features(
        original_crop_bgr, fruit_mask=fruit_mask
    )

    result = classify_tomato(result)

    return result, diagnostics


def save_diagnostics(out_dir: str, diagnostics: Dict[str, np.ndarray]) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)

    for name in (
        "fruit_mask",
        "green_mask",
        "turning_mask",
        "pink_mask",
        "mature_red_mask",
        "colour_break_mask",
        "red_orange_mask",
        "other_mask",
        "lab_l_channel",
    ):
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
    DEFAULT_DEBUG_DIR = SCRIPT_DIR / "tomato_full_test_debug"

    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="Original-colour tomato crop/image")
    parser.add_argument("--mask", help="Optional YOLO binary tomato mask")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_DEBUG_DIR),
        help="Default: fruit_ripeness/tomato_ripeness/tomato_full_test_debug/",
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

    result, diagnostics = assess_tomato_ripeness(img, fruit_mask=mask)

    print("\nTOMATO RIPENESS RESULT")
    print("----------------------")
    print(f"YOLO ROI pixels       : {result.total_fruit_pixels}")
    print(f"Green pixels          : {result.green_pixels}")
    print(f"Turning pixels        : {result.turning_pixels}")
    print(f"Pink pixels           : {result.pink_pixels}")
    print(f"Mature/red pixels     : {result.mature_red_pixels}")
    print(
        f"Colour-break pixels   : {result.colour_break_pixels} (Turning + Pink + Mature/red)"
    )
    print(f"Red-orange pixels     : {result.red_orange_pixels} (Pink + Mature/red)")
    print(f"Green coverage        : {result.green_pct:.2f}%")
    print(f"Colour-break coverage : {result.colour_break_pct:.2f}%")
    print(f"Red-orange coverage   : {result.red_orange_pct:.2f}%")
    print(f"Other coverage        : {result.other_pct:.2f}%")

    if result.median_lstar_red_orange is not None:
        print(f"Mean L* mature-red    : {result.mean_lstar_red_orange:.2f}")
        print(f"Median L* mature-red  : {result.median_lstar_red_orange:.2f}")
        print(f"L* Index 5/6 threshold: {DARK_LSTAR_THRESHOLD:.2f}")
    else:
        print("Mature-red L*         : N/A")

    print(f"Prediction            : {result.maturity_label}")
    print(f"Rule basis            : {result.rule_basis}")

    save_diagnostics(args.out, diagnostics)

    print("\nDiagnostic masks saved to:", os.path.abspath(args.out))
