"""
Stage 1 - Preprocessing.

Photometric conditioning only: 1.1 white balance, 1.2 CLAHE.
No geometric change, so masks from Stage 2 still index into the original image.
"""

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np


@dataclass
class PreprocessResult:
    bgr: np.ndarray
    steps: dict = field(default_factory=dict)   # intermediates, only when keep_steps


@dataclass
class PreprocessParams:
    enabled: bool = True

    white_balance_enabled: bool = False
    wb_strength: float = 0.5            # 0 = no change, 1 = full grey-world

    clahe_enabled: bool = True
    clahe_clip_limit: float = 1.0       # CLAHE strength
    clahe_tile_grid: int = 8            # NxN tiles; 8 is the usual default
    clahe_contrast_gate: float = 55.0   # skip CLAHE when L.std() is above this


PARAMS = PreprocessParams()


def _grayworld(image, saturation_threshold):
    """Grey-world in numpy; fallback when cv.xphoto is unavailable."""
    src = image.astype(np.float32)
    high = src.max(axis=2)
    low = src.min(axis=2)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0)
    keep = saturation <= saturation_threshold
    if not keep.any():
        keep = np.ones_like(saturation, dtype=bool)
    means = src[keep].reshape(-1, 3).mean(axis=0) + 1e-6
    return np.clip(src * (means.mean() / means), 0, 255).astype(np.uint8)


def white_balance(image, saturation_threshold=0.9,
                  strength=0.5, max_gain=1.6):
    """Remove the lighting colour cast (gain-capped, blended grey-world)."""
    try:
        wb = cv.xphoto.createGrayworldWB()
        wb.setSaturationThreshold(saturation_threshold)
        balanced = wb.balanceWhite(image)
    except Exception:
        balanced = _grayworld(image, saturation_threshold)

    # Cap the gain and blend back: plain grey-world pushes a single vivid fruit
    # toward its complement (a red tomato turns blue).
    src = image.astype(np.float32)
    dst = balanced.astype(np.float32)
    eps = 1e-6
    src_mean = src.reshape(-1, 3).mean(axis=0) + eps
    dst_mean = dst.reshape(-1, 3).mean(axis=0) + eps
    gains = np.clip(dst_mean / src_mean, 1.0 / max_gain, max_gain)
    capped = np.clip(src * gains, 0, 255)

    blended = cv.addWeighted(capped, strength, src, 1.0 - strength, 0)
    return blended.astype(np.uint8)


def apply_clahe(image, clip_limit=1.0, tile_grid_size=(8, 8), contrast_gate=55.0):
    """Even out local brightness. Returns (image, was_applied)."""
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    L, A, B = cv.split(lab)

    # Skipped when contrast is already high, to avoid adding artificial texture.
    # Return the input untouched: the LAB round trip alone would shift pixels.
    if float(L.std()) >= contrast_gate:
        return image, False

    # L only - equalising the colour channels would shift hue, which ripeness reads.
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L = clahe.apply(L)
    return cv.cvtColor(cv.merge([L, A, B]), cv.COLOR_LAB2BGR), True


def preprocess(image, keep_steps=False, params=None):
    """Run Stage 1 and return the corrected frame."""
    p = params if params is not None else PARAMS
    steps = {}

    if not p.enabled:
        return PreprocessResult(bgr=image.copy(), steps=steps)

    img = image.copy()

    # 1.1
    if p.white_balance_enabled:
        img = white_balance(img, strength=p.wb_strength)
        if keep_steps:
            steps["1.1_white_balanced"] = img.copy()

    # 1.2
    if p.clahe_enabled:
        img, clahe_applied = apply_clahe(
            img, p.clahe_clip_limit,
            (p.clahe_tile_grid, p.clahe_tile_grid),
            p.clahe_contrast_gate)
        if keep_steps:
            steps["1.2_clahe"] = img.copy()
            steps["1.2_clahe_applied"] = clahe_applied

    return PreprocessResult(bgr=img, steps=steps)


# CLI test:  python preprocessing.py <input> [output]
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <input_image> [output_image]")
        sys.exit(1)

    src_path = sys.argv[1]
    dst_path = sys.argv[2] if len(sys.argv) > 2 else "preprocessed.png"

    src = cv.imread(src_path)
    if src is None:
        print(f"Could not read image: {src_path}")
        sys.exit(1)

    result = preprocess(src)
    cv.imwrite(dst_path, result.bgr)

    print(f"Input : {src.shape[1]}x{src.shape[0]}")
    print(f"Output: {result.bgr.shape[1]}x{result.bgr.shape[0]}")
    print(f"Saved final image to: {dst_path}")
