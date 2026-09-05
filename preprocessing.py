"""
Stage 1 - Preprocessing module for the Fruit Ripeness Assessment system.

The purpose of this stage is to remove nouses (lighting colour, sensor noise, uneven brightness),
so that the later stages measure the fruit itself.

Pipeline order (colour of light -> noise -> local contrast -> crop -> package):
    1.1 white_balance             remove the colour cast of the lighting
    1.2 denoise                   remove speckle/grain, keep edges sharp
    1.3 apply_clahe               even out local brightness (conditional)
    1.4 otsu_cropping             isolate each object as its own crop
    1.5 preprocess (packaging)    cache BGR + HSV + LAB for downstream stages


"""

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np


# ----------------------------------------------------------------------
# Result container (what Stage 1 hands to Stage 2)
# ----------------------------------------------------------------------
@dataclass
class PreprocessResult:
    bgr: np.ndarray                 # final cleaned image (BGR)
    hsv: np.ndarray                 # cached HSV conversion (used by segmentation/ripeness)
    lab: np.ndarray                 # cached LAB conversion
    crops: list
    crop_rects: list
    steps: dict = field(default_factory=dict)   # intermediate images, for debugging/report figures


# ----------------------------------------------------------------------
# Tunable parameters (edited live by the GUI)
# ----------------------------------------------------------------------
@dataclass
class PreprocessParams:
    enabled: bool = True

    white_balance_enabled: bool = False
    wb_saturation_threshold: float = 0.9
    wb_strength: float = 0.5
    wb_max_gain: float = 1.6

    denoise_enabled: bool = False
    median_ksize: int = 3
    bilateral_d: int = 2
    bilateral_sigma_colour: int = 25
    bilateral_sigma_space: int = 25

    clahe_enabled: bool = True
    clahe_clip_limit: float = 1.0
    clahe_tile_grid: int = 8
    clahe_contrast_gate: float = 55.0

    otsu_min_area: int = 2000
    otsu_pad_percent: float = 0.20
    otsu_blur_ksize: int = 11
    otsu_morph_ksize: int = 7
    otsu_dilate_ksize: int = 5


PARAMS = PreprocessParams()


def _odd(value, minimum=1):
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


# ----------------------------------------------------------------------
# 1.2  White balance (saturation-thresholded grey-world)
# ----------------------------------------------------------------------
def _grayworld(image, saturation_threshold):
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
    """Neutralise the colour cast introduced by the lighting, gently.

    Plain grey-world assumes the scene averages to grey. That breaks when a
    single vivid fruit fills the frame (e.g. a red tomato): grey-world then adds
    the opposite colour (blue) to force the average grey, discolouring the fruit
    and confusing the classifier. Two safeguards prevent that:

    - ``max_gain`` caps how far any channel may be scaled, so a colour-dominated
      image can't be pushed hard toward its complement.
    - ``strength`` blends the corrected image with the original (0 = no change,
      1 = full grey-world). At 0.5 a real lighting cast is still reduced, but a
      naturally saturated fruit is left close to its true colour.

    The saturation threshold still excludes vivid pixels when estimating gains.
    """
    try:
        wb = cv.xphoto.createGrayworldWB()
        wb.setSaturationThreshold(saturation_threshold)
        balanced = wb.balanceWhite(image)
    except Exception:
        # opencv-contrib unavailable or incompatible; equivalent grey-world in numpy.
        balanced = _grayworld(image, saturation_threshold)

    # Limit per-channel gain so the correction can't over-shift a colour cast.
    src = image.astype(np.float32)
    dst = balanced.astype(np.float32)
    eps = 1e-6
    src_mean = src.reshape(-1, 3).mean(axis=0) + eps
    dst_mean = dst.reshape(-1, 3).mean(axis=0) + eps
    gains = np.clip(dst_mean / src_mean, 1.0 / max_gain, max_gain)
    capped = np.clip(src * gains, 0, 255)

    # Blend the (gain-limited) correction with the original at `strength`.
    blended = cv.addWeighted(capped, strength, src, 1.0 - strength, 0)
    return blended.astype(np.uint8)


# ----------------------------------------------------------------------
# 1.3  Denoise (edge-preserving)
# ----------------------------------------------------------------------
def denoise(image, median_ksize=3, bilateral_d=2,
            bilateral_sigma_colour=25   , bilateral_sigma_space=25):
    """Remove speckle and grain while keeping the fruit boundary sharp.

    Median blur first: highly effective against salt-and-pepper speckle.
    Bilateral filter second: smooths flat regions but preserves edges (unlike a
    plain Gaussian blur, which would soften the outline the next stages rely on).
    """
    speckle_free = cv.medianBlur(image, _odd(median_ksize, 1))
    smoothed = cv.bilateralFilter(
        speckle_free, bilateral_d, bilateral_sigma_colour, bilateral_sigma_space
    )
    return smoothed


# ----------------------------------------------------------------------
# 1.4  CLAHE on the L channel (conditional)
# ----------------------------------------------------------------------
def apply_clahe(image, clip_limit=1.0, tile_grid_size=(8, 8), contrast_gate=55.0):
    """Even out local brightness, but only on the L (lightness) channel and only
    when the image actually needs it.

    - L channel only: applying CLAHE to colour channels would distort hue, which
      ripeness assessment depends on.
    - Conditional: if global contrast (std-dev of L) is already high, CLAHE is
      skipped to avoid adding artificial texture and wasted computation.

    Returns (result_image, was_applied) so the caller can log the decision.
    """
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    L, A, B = cv.split(lab)

    global_contrast = float(L.std())
    was_applied = global_contrast < contrast_gate

    if was_applied:
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        L = clahe.apply(L)

    result = cv.cvtColor(cv.merge([L, A, B]), cv.COLOR_LAB2BGR)
    return result, was_applied


def otsu_cropping(image, min_area=2000, pad_percent=0.20,
                  blur_ksize=11, morph_ksize=7, dilate_ksize=5):
    if image is None:
        print("Empty image provided to Otsu function.")
        return [], []

    blur_ksize = _odd(blur_ksize, 1)
    morph_ksize = max(1, int(morph_ksize))
    dilate_ksize = max(1, int(dilate_ksize))

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    _, S, _ = cv.split(hsv)
    blurred_sv = cv.GaussianBlur(S, (blur_ksize, blur_ksize), 0)

    _, otsu_mask = cv.threshold(blurred_sv, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    clean_kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
    clean_mask = cv.morphologyEx(otsu_mask, cv.MORPH_CLOSE, clean_kernel, iterations=1)
    clean_mask = cv.morphologyEx(clean_mask, cv.MORPH_OPEN, clean_kernel, iterations=1)
    extended_mask = cv.dilate(clean_mask, np.ones((dilate_ksize, dilate_ksize), np.uint8), iterations=1)

    contours, _ = cv.findContours(extended_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    H, W = image.shape[:2]
    crops, rects = [], []

    for contour in sorted(contours, key=lambda c: cv.boundingRect(c)[0]):
        if cv.contourArea(contour) < min_area:
            continue

        x, y, w, h = cv.boundingRect(contour)
        pad_x = int(w * pad_percent)
        pad_y = int(h * pad_percent)

        new_x = max(0, x - pad_x)
        new_y = max(0, y - pad_y)
        new_w = min(W - new_x, w + (pad_x * 2))
        new_h = min(H - new_y, h + (pad_y * 2))

        crops.append(image[new_y:new_y+new_h, new_x:new_x+new_w])
        rects.append((new_x, new_y, new_w, new_h))

    return crops, rects

# ----------------------------------------------------------------------
# 1.5  Full pipeline (packages BGR + HSV + LAB for downstream stages)
# ----------------------------------------------------------------------
def preprocess(image, keep_steps=False, params=None):

    p = params if params is not None else PARAMS
    steps = {}

    if not p.enabled:
        return PreprocessResult(
            bgr=image.copy(),
            hsv=cv.cvtColor(image, cv.COLOR_BGR2HSV),
            lab=cv.cvtColor(image, cv.COLOR_BGR2LAB),
            crops=[],
            crop_rects=[],
            steps=steps,
        )

    img = image.copy()

    # 1.1
    if p.white_balance_enabled:
        img = white_balance(img, p.wb_saturation_threshold, p.wb_strength, p.wb_max_gain)
        if keep_steps:
            steps["1.1_white_balanced"] = img.copy()

    # 1.2
    if p.denoise_enabled:
        img = denoise(img, _odd(p.median_ksize, 1), p.bilateral_d,
                      p.bilateral_sigma_colour, p.bilateral_sigma_space)
        if keep_steps:
            steps["1.2_denoised"] = img.copy()

    # 1.3
    if p.clahe_enabled:
        img, clahe_applied = apply_clahe(
            img, p.clahe_clip_limit,
            (p.clahe_tile_grid, p.clahe_tile_grid),
            p.clahe_contrast_gate)
        if keep_steps:
            steps["1.3_clahe"] = img.copy()
            steps["1.3_clahe_applied"] = clahe_applied

    # 1.4
    crops, crop_rects = otsu_cropping(
        img, p.otsu_min_area, p.otsu_pad_percent, p.otsu_blur_ksize,
        p.otsu_morph_ksize, p.otsu_dilate_ksize)
    if keep_steps:
        steps["1.4_crops"] = crops
        steps["1.4_crop_rects"] = crop_rects

    # 1.5  cache colour spaces once
    final_bgr = img
    final_hsv = cv.cvtColor(final_bgr, cv.COLOR_BGR2HSV)
    final_lab = cv.cvtColor(final_bgr, cv.COLOR_BGR2LAB)

    return PreprocessResult(
        bgr=final_bgr,
        hsv=final_hsv,
        lab=final_lab,
        crops=crops,
        crop_rects=crop_rects,
        steps=steps,
    )


# ----------------------------------------------------------------------
# Simple command-line test:  python preprocessing.py <input> <output>
# ----------------------------------------------------------------------
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

    result = preprocess(src, keep_steps=True)
    cv.imwrite(dst_path, result.bgr)

    print(f"Input : {src.shape[1]}x{src.shape[0]}")
    print(f"Output: {result.bgr.shape[1]}x{result.bgr.shape[0]}")
    print(f"Saved final image to: {dst_path}")