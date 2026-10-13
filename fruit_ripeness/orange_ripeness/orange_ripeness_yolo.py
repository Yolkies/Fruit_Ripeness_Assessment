from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

INDEX2_MIN = 20.0
INDEX3_MIN = 50.0
INDEX4_MIN = 70.0
INDEX5_MIN = 90.0

CCI_EPSILON = 1e-6


@dataclass
class OrangeRipenessResult:
    maturity_index: int
    maturity_label: str
    rule_basis: str

    total_fruit_pixels: int

    yellow_orange_pixels: int
    non_yellow_orange_pixels: int
    undefined_cci_pixels: int

    yellow_orange_pct: float
    non_yellow_orange_pct: float
    undefined_cci_pct: float

    mean_cci: Optional[float]
    median_cci: Optional[float]


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
        raise ValueError("Input orange crop is empty")

    non_black = np.any(masked_crop_bgr != 0, axis=2)
    return non_black.astype(np.uint8) * 255


def prepare_orange_for_colour_analysis(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    if original_crop_bgr is None or original_crop_bgr.size == 0:
        raise ValueError("Input orange crop is empty")

    h, w = original_crop_bgr.shape[:2]

    if fruit_mask is None:
        fruit_mask = fruit_mask_from_black_background(original_crop_bgr)
    else:
        fruit_mask = _normalise_binary_mask(fruit_mask, (h, w))

    bgr = original_crop_bgr.copy()
    lab_opencv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    return {"bgr": bgr, "lab_opencv": lab_opencv, "fruit_mask": fruit_mask}


def _opencv_lab_to_cielab(
    lab_opencv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = lab_opencv.astype(np.float32)

    lstar = lab[:, :, 0] * (100.0 / 255.0)
    astar = lab[:, :, 1] - 128.0
    bstar = lab[:, :, 2] - 128.0

    return lstar, astar, bstar


def calculate_cci_map(lab_opencv: np.ndarray, fruit_mask: np.ndarray):
    lstar, astar, bstar = _opencv_lab_to_cielab(lab_opencv)

    roi_bool = fruit_mask > 0
    denominator = lstar * bstar

    valid_bool = (
        roi_bool & np.isfinite(denominator) & (np.abs(denominator) > CCI_EPSILON)
    )

    cci_map = np.full(lstar.shape, np.nan, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        cci_map[valid_bool] = 1000.0 * astar[valid_bool] / denominator[valid_bool]

    yellow_orange_bool = valid_bool & (cci_map >= 0.0)
    non_yellow_orange_bool = valid_bool & (cci_map < 0.0)
    valid_cci_mask = valid_bool.astype(np.uint8) * 255
    yellow_orange_mask = yellow_orange_bool.astype(np.uint8) * 255
    non_yellow_orange_mask = non_yellow_orange_bool.astype(np.uint8) * 255

    return (cci_map, valid_cci_mask, yellow_orange_mask, non_yellow_orange_mask)


def _make_cci_visualization(cci_map: np.ndarray, fruit_mask: np.ndarray) -> np.ndarray:
    vis = np.zeros(cci_map.shape, dtype=np.uint8)

    valid = (fruit_mask > 0) & np.isfinite(cci_map)

    if not np.any(valid):
        return vis

    values = cci_map[valid]

    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))

    if high <= low:
        high = low + 1.0

    clipped = np.clip(cci_map, low, high)
    scaled = (clipped - low) / (high - low) * 255.0

    vis[valid] = scaled[valid].astype(np.uint8)

    return vis


def measure_orange_features(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
):
    data = prepare_orange_for_colour_analysis(original_crop_bgr, fruit_mask=fruit_mask)

    lab_opencv = data["lab_opencv"]
    roi_mask = data["fruit_mask"]

    total = int(cv2.countNonZero(roi_mask))

    if total == 0:
        raise ValueError("YOLO orange mask contains no fruit pixels")

    cci_map, valid_cci_mask, yellow_orange_mask, non_yellow_orange_mask = (
        calculate_cci_map(lab_opencv, roi_mask)
    )

    yellow_orange_pixels = int(cv2.countNonZero(yellow_orange_mask))
    non_yellow_orange_pixels = int(cv2.countNonZero(non_yellow_orange_mask))
    valid_pixels = int(cv2.countNonZero(valid_cci_mask))
    undefined_cci_pixels = total - valid_pixels
    yellow_orange_pct = 100.0 * yellow_orange_pixels / total
    non_yellow_orange_pct = 100.0 * non_yellow_orange_pixels / total
    undefined_cci_pct = 100.0 * undefined_cci_pixels / total

    valid_values = cci_map[(roi_mask > 0) & np.isfinite(cci_map)]

    mean_cci = None
    median_cci = None

    if valid_values.size > 0:
        mean_cci = float(np.mean(valid_values))
        median_cci = float(np.median(valid_values))

    undefined_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(valid_cci_mask))

    cci_visualization = _make_cci_visualization(cci_map, roi_mask)

    result = OrangeRipenessResult(
        maturity_index=0,
        maturity_label="Not classified",
        rule_basis="",
        total_fruit_pixels=total,
        yellow_orange_pixels=yellow_orange_pixels,
        non_yellow_orange_pixels=non_yellow_orange_pixels,
        undefined_cci_pixels=undefined_cci_pixels,
        yellow_orange_pct=yellow_orange_pct,
        non_yellow_orange_pct=non_yellow_orange_pct,
        undefined_cci_pct=undefined_cci_pct,
        mean_cci=mean_cci,
        median_cci=median_cci,
    )

    diagnostics = {
        "bgr": data["bgr"],
        "fruit_mask": roi_mask,
        "yellow_orange_mask": yellow_orange_mask,
        "non_yellow_orange_mask": non_yellow_orange_mask,
        "undefined_cci_mask": undefined_mask,
        "cci_visualization": cci_visualization,
        "lab_l_channel": lab_opencv[:, :, 0],
        "lab_a_channel": lab_opencv[:, :, 1],
        "lab_b_channel": lab_opencv[:, :, 2],
    }

    return result, diagnostics


def classify_orange(measurement: OrangeRipenessResult) -> OrangeRipenessResult:
    coverage = measurement.yellow_orange_pct

    if coverage < INDEX2_MIN:
        measurement.maturity_index = 1
        measurement.maturity_label = "Index 1 - Green Fruit"
        measurement.rule_basis = f"Yellow/orange coverage={coverage:.2f}% (<20%)."

    elif coverage < INDEX3_MIN:
        measurement.maturity_index = 2
        measurement.maturity_label = "Index 2 - Color Transition"
        measurement.rule_basis = (
            f"Yellow/orange coverage={coverage:.2f}% (>=20% and <50%)."
        )

    elif coverage < INDEX4_MIN:
        measurement.maturity_index = 3
        measurement.maturity_label = "Index 3 - Early Ripe"
        measurement.rule_basis = (
            f"Yellow/orange coverage={coverage:.2f}% (>=50% and <70%)."
        )

    elif coverage < INDEX5_MIN:
        measurement.maturity_index = 4
        measurement.maturity_label = "Index 4 - Moderately Ripe"
        measurement.rule_basis = (
            f"Yellow/orange coverage={coverage:.2f}% (>=70% and <90%)."
        )

    else:
        measurement.maturity_index = 5
        measurement.maturity_label = "Index 5 - Fully Ripe"
        measurement.rule_basis = f"Yellow/orange coverage={coverage:.2f}% (>=90%)."

    return measurement


def assess_orange_ripeness(
    original_crop_bgr: np.ndarray, fruit_mask: Optional[np.ndarray] = None
):
    result, diagnostics = measure_orange_features(
        original_crop_bgr, fruit_mask=fruit_mask
    )

    result = classify_orange(result)

    return result, diagnostics


def save_diagnostics(out_dir: str, diagnostics: Dict[str, np.ndarray]) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)

    for name in (
        "fruit_mask",
        "yellow_orange_mask",
        "non_yellow_orange_mask",
        "undefined_cci_mask",
        "cci_visualization",
        "lab_l_channel",
        "lab_a_channel",
        "lab_b_channel",
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
    DEFAULT_DEBUG_DIR = SCRIPT_DIR / "orange_full_test_debug"

    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="Original-colour orange crop/image")
    parser.add_argument("--mask", help="Optional YOLO binary orange mask")

    parser.add_argument(
        "--out",
        default=str(DEFAULT_DEBUG_DIR),
        help="Default: fruit_ripeness/orange_ripeness/orange_full_test_debug/",
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

    result, diagnostics = assess_orange_ripeness(img, fruit_mask=mask)

    print("\nORANGE RIPENESS RESULT")
    print("-----------------------")
    print(f"Fruit pixels               : {result.total_fruit_pixels}")
    print(f"Yellow/orange pixels       : {result.yellow_orange_pixels} (CCI >= 0)")
    print(f"Non-yellow/orange pixels   : {result.non_yellow_orange_pixels} (CCI < 0)")
    print(f"Undefined CCI pixels       : {result.undefined_cci_pixels}")
    print(f"Yellow/orange coverage     : {result.yellow_orange_pct:.2f}%")
    print(f"Non-yellow/orange coverage : {result.non_yellow_orange_pct:.2f}%")
    print(f"Undefined CCI coverage     : {result.undefined_cci_pct:.2f}%")

    if result.mean_cci is not None:
        print(f"Mean CCI                   : {result.mean_cci:.2f}")
        print(f"Median CCI                 : {result.median_cci:.2f}")
    else:
        print("Mean CCI                   : N/A")
        print("Median CCI                 : N/A")

    print(f"Prediction                 : {result.maturity_label}")
    print(f"Rule basis                 : {result.rule_basis}")

    save_diagnostics(args.out, diagnostics)

    print("\nDiagnostic masks saved to:", os.path.abspath(args.out))
