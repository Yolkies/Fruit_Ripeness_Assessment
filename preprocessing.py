"""
Stage 1 - Preprocessing module for the Fruit Ripeness Assessment system.

The purpose of this stage is to remove nouses (lighting colour, sensor noise, glare, uneven brightness),
so that the later stages measure the fruit itself.

Pipeline order (order matters: geometry -> colour of light -> noise -> local
contrast -> glare -> package):
    1.1 resize_normalise          normalise image size
    1.2 white_balance             remove the colour cast of the lighting
    1.3 denoise                   remove speckle/grain, keep edges sharp
    1.4 apply_clahe               even out local brightness (conditional)
    1.5 remove_specular_highlights  patch shiny glare spots
    1.6 preprocess (packaging)    cache BGR + HSV + LAB for downstream stages


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
    scale: float                    # resize factor, for mapping measurements back to source
    crops: list
    crop_rects: list
    steps: dict = field(default_factory=dict)   # intermediate images, for debugging/report figures


# ----------------------------------------------------------------------
# Tunable parameters (edited live by the GUI)
# ----------------------------------------------------------------------
@dataclass
class PreprocessParams:
    resize_enabled: bool = False
    target_long_side: int = 800

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

    glare_enabled: bool = False
    glare_v_min: int = 240
    glare_s_max: int = 90
    glare_inpaint_radius: int = 3

    otsu_min_area: int = 2000
    otsu_pad_percent: float = 0.20
    otsu_blur_ksize: int = 11
    otsu_morph_ksize: int = 7
    otsu_dilate_ksize: int = 5


PARAMS = PreprocessParams()


# ----------------------------------------------------------------------
# 1.1  Resize / normalise size
# ----------------------------------------------------------------------
def resize_normalise(image, target_long_side=800):
    """Scale the image so its longest side equals ``target_long_side`` px.

    Fixing the size first means every later kernel (a 5 px blur, an 8x8 CLAHE
    tile) behaves consistently across phone photos and webcam frames. The scale
    factor is returned so pixel measurements can be mapped back to the original
    resolution (Image Calibration requirement).
    """
    h, w = image.shape[:2]
    scale = target_long_side / max(h, w)
    # INTER_AREA is best for shrinking; INTER_CUBIC is best for enlarging.
    interp = cv.INTER_AREA if scale < 1 else cv.INTER_CUBIC
    resized = cv.resize(image, None, fx=scale, fy=scale, interpolation=interp)
    return resized, scale


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
    if hasattr(cv, "xphoto"):
        wb = cv.xphoto.createGrayworldWB()
        wb.setSaturationThreshold(saturation_threshold)
        balanced = wb.balanceWhite(image)
    else:
        # opencv-contrib is not installed; equivalent grey-world in numpy.
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
    speckle_free = cv.medianBlur(image, median_ksize)
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


# ----------------------------------------------------------------------
# 1.5  Specular highlight (glare) suppression
# ----------------------------------------------------------------------
def remove_specular_highlights(image, v_min=240, s_max=90, inpaint_radius=3):
    """Detect small bright glare spots (high brightness, low saturation) and
    inpaint them using surrounding skin colour.

    Glare carries no real colour information and can be mistaken for a blemish
    or punch holes in the segmentation mask, so it is patched out. Returns
    (result_image, glare_mask) so the mask can be shown in report figures.
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    _, S, V = cv.split(hsv)

    glare_mask = ((V > v_min) & (S < s_max)).astype(np.uint8) * 255
    # Dilate a little so the inpaint also covers the soft edge around each spot.
    glare_mask = cv.dilate(glare_mask, np.ones((3, 3), np.uint8), iterations=2)

    if cv.countNonZero(glare_mask) == 0:
        return image, glare_mask   # nothing to do (e.g. soft, diffuse lighting)

    result = cv.inpaint(image, glare_mask, inpaint_radius, cv.INPAINT_TELEA)
    return result, glare_mask


def _odd(value, minimum=1):
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


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
# 1.6  Full pipeline (packages BGR + HSV + LAB for downstream stages)
# ----------------------------------------------------------------------
def preprocess(image, target_long_side=None, keep_steps=False, params=None):

    p = params if params is not None else PARAMS
    steps = {}

    # 1.1
    long_side = p.target_long_side if target_long_side is None else target_long_side
    img2, scale = resize_normalise(image, long_side)
    if keep_steps:
        steps["1.1_resized"] = img2.copy()

    img = img2.copy() if p.resize_enabled else image.copy()

    # 1.2
    if p.white_balance_enabled:
        img = white_balance(img, p.wb_saturation_threshold, p.wb_strength, p.wb_max_gain)
        if keep_steps:
            steps["1.2_white_balanced"] = img.copy()

    # 1.3
    if p.denoise_enabled:
        img = denoise(img, _odd(p.median_ksize, 1), p.bilateral_d,
                      p.bilateral_sigma_colour, p.bilateral_sigma_space)
        if keep_steps:
            steps["1.3_denoised"] = img.copy()

    # 1.4
    if p.clahe_enabled:
        img, clahe_applied = apply_clahe(
            img, p.clahe_clip_limit,
            (p.clahe_tile_grid, p.clahe_tile_grid),
            p.clahe_contrast_gate)
        if keep_steps:
            steps["1.4_clahe"] = img.copy()
            steps["1.4_clahe_applied"] = clahe_applied

    # 1.5
    if p.glare_enabled:
        img, glare_mask = remove_specular_highlights(
            img, p.glare_v_min, p.glare_s_max, p.glare_inpaint_radius)
        if keep_steps:
            steps["1.5_glare_mask"] = glare_mask
            steps["1.5_deglared"] = img.copy()

    #otsu
    crops, crop_rects = otsu_cropping(
        img, p.otsu_min_area, p.otsu_pad_percent,
        p.otsu_blur_ksize, p.otsu_morph_ksize, p.otsu_dilate_ksize)
    if keep_steps:
        steps["1.6_crops"] = crops
        steps["1.6_crop_rects"] = crop_rects

    # 1.6  cache colour spaces once
    final_bgr = img
    final_hsv = cv.cvtColor(final_bgr, cv.COLOR_BGR2HSV)
    final_lab = cv.cvtColor(final_bgr, cv.COLOR_BGR2LAB)

    return PreprocessResult(
        bgr=final_bgr,
        hsv=final_hsv,
        lab=final_lab,
        scale=scale,
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
    print(f"Output: {result.bgr.shape[1]}x{result.bgr.shape[0]} (scale {result.scale:.3f})")
    print(f"Saved final image to: {dst_path}")