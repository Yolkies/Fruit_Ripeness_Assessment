"""
Stage 1: Image Preprocessing Pipeline

This module implements the "Preprocessing" and "Image Calibration" shared core
functional requirements from the assignment specification:

    * Preprocessing   : noise removal + image enhancement
                        (Gaussian / Median / Bilateral filtering, contrast stretching, CLAHE)
    * Calibration     : spatial scaling to a consistent size (aspect-ratio preserved)
    * Fruit isolation : HSV + Otsu + morphology segmentation to separate the fruit
                        from its background, ready for feature extraction later.

Design note for FRUIT RIPENESS specifically:
    Ripeness is judged almost entirely from SURFACE COLOUR (green -> yellow -> red,
    plus brown/black blemishes). So every enhancement step here is chosen to CLEAN
    the image WITHOUT destroying colour information:
        - denoising uses colour-aware / edge-preserving filters
        - contrast is enhanced on the L (lightness) channel of LAB, leaving colour intact
          (a naive grayscale histogram-equalisation would throw the colour away)
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TARGET_SIZE = 256                      # calibrated square size (pixels)
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FRUIT_CLASSES = ["Apple", "Banana", "Mango", "Orange", "Tomato"]


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------
def load_image(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


# ----------------------------------------------------------------------------
# 2. Calibration  ->  spatial scaling to a consistent size
# ----------------------------------------------------------------------------
def resize_keep_aspect(img, size=TARGET_SIZE, pad_colour=(0, 0, 0)):
    """
    Resize so the longest side == `size`, then pad the shorter side so the
    output is a square `size x size` image.  Padding (instead of a plain
    cv.resize to a square) keeps the fruit from being stretched/distorted,
    which matters because a squashed fruit changes its apparent shape.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

    # Centre the resized image on a square canvas.
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left
    return cv.copyMakeBorder(resized, top, bottom, left, right,
                             cv.BORDER_CONSTANT, value=pad_colour)


# ----------------------------------------------------------------------------
# 3. Noise removal
# ----------------------------------------------------------------------------
def denoise(img, method="bilateral"):
    """
    Remove noise while trying to preserve the colour edges that ripeness
    detection depends on.

    method:
        "gaussian"  - cv.GaussianBlur  : fast, good for Gaussian noise, blurs edges.
        "median"    - cv.medianBlur    : best for salt-and-pepper speckle.
        "bilateral" - cv.bilateralFilter: edge-preserving (DEFAULT, recommended).
        "nlmeans"   - cv.fastNlMeansDenoisingColored: strongest, slower.

    """
    if method == "gaussian":
        return cv.GaussianBlur(img, (5, 5), 0)
    if method == "median":
        return cv.medianBlur(img, 5)
    if method == "bilateral":
        # d=9 neighbourhood, sigmaColor=75, sigmaSpace=75 (handbook values).
        return cv.bilateralFilter(img, 9, 75, 75)
    if method == "nlmeans":
        return cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    raise ValueError(f"Unknown denoise method: {method}")


# ----------------------------------------------------------------------------
# 4. Contrast / illumination enhancement
# ----------------------------------------------------------------------------
def enhance_contrast(img, method="clahe"):
    """
    Improve contrast so ripeness cues stand out under uneven lighting.

    method:
        "clahe"    - Contrast Limited Adaptive Histogram Equalisation applied to
                     the L channel of LAB.  This is the colour-SAFE version of the
                     histogram equalisation covered in the handbook: it fixes
                     lightness/contrast while leaving the a,b colour channels
                     untouched, so the fruit's hue is preserved.
        "stretch"  - Contrast stretching (min-max normalisation).  Explicitly
                     named in the assignment spec.  Applied per-channel in LAB's
                     L channel to avoid colour shifts.

    """
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    L, a, b = cv.split(lab)

    if method == "clahe":
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
    elif method == "stretch":
        # Linear contrast stretch: remap [min,max] -> [0,255].
        lo, hi = int(L.min()), int(L.max())
        if hi > lo:
            L = ((L.astype(np.float32) - lo) * 255.0 / (hi - lo)).astype(np.uint8)
    else:
        raise ValueError(f"Unknown contrast method: {method}")

    merged = cv.merge((L, a, b))
    return cv.cvtColor(merged, cv.COLOR_LAB2BGR)


# ----------------------------------------------------------------------------
# 5. Fruit segmentation  ->  isolate the fruit from the background
# ----------------------------------------------------------------------------
def _shape_score(contour):
    """
    Score a contour by how FRUIT-SHAPED it is, not just how big it is.

        circularity = 4*pi*area / perimeter^2   (1.0 = perfect circle,
                       low for jagged/branching shapes like leaf clusters)
        solidity    = area / convex_hull_area   (1.0 = no concave dents,
                       low when the shape has lobes/appendages sticking out,
                       which is exactly what a leaf or stem looks like)

    Returns (area, circularity, solidity, score) where score rewards a region
    that is BOTH large AND fruit-shaped, so a big leafy blob is out-scored by
    a smaller, cleanly round fruit blob.
    """
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    hull_area = cv.contourArea(cv.convexHull(contour))
    solidity = (area / hull_area) if hull_area > 0 else 0.0
    score = area * circularity * solidity
    return area, circularity, solidity, score


def _select_best_blob(mask, min_area_frac=0.01):
    """
    Among all connected white regions in `mask`, keep only the one that scores
    best on _shape_score.  This is the key fix for leaf/stem contamination:
    picking the most FRUIT-SHAPED region instead of always picking the LARGEST
    region means a big tangle of leaves loses out to a smaller, round fruit.

    Also returns the winning contour's (circularity, solidity) so the caller
    can flag images where even the best candidate doesn't look very fruit-like
    (a sign that fruit and leaf are too tightly fused to separate by shape).
    """
    h, w = mask.shape[:2]
    min_area = min_area_frac * h * w
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best_contour, best_score, best_metrics = None, -1.0, (0.0, 0.0)
    for c in contours:
        area, circ, solid, score = _shape_score(c)
        if area < min_area:
            continue
        if score > best_score:
            best_contour, best_score, best_metrics = c, score, (circ, solid)

    clean = np.zeros_like(mask)
    if best_contour is not None:
        cv.drawContours(clean, [best_contour], -1, 255, thickness=cv.FILLED)
    return clean, best_metrics


def _strip_minor_green(mask, hue, green_dominance_thresh=0.4):
    """
    Remove green (leaf/stem) pixels from the mask WHEN they are clearly a minor
    attachment rather than the fruit's own colour.

    Why this is safe: a leaf or stem touching the fruit contributes a small
    slice of green to an otherwise orange/red/yellow blob.  A genuinely UNRIPE
    fruit (green banana, green mango, green tomato) is green all over -- green
    is the majority of the blob, not a minor patch.  So the rule is: only strip
    green pixels if they make up LESS than `green_dominance_thresh` of the
    fruit's own mask.  If green pixels dominate, this function assumes that IS
    the fruit's ripeness colour and leaves it alone.

    A plain per-pixel hue cut leaves a ragged, pixel-by-pixel boundary where
    the leaf used to be (individual flipped pixels inflate the perimeter), so
    a light open+close smooths that edge before re-selecting the best blob.

    Returns (mask, green_fraction) so the caller can log/inspect what was cut.
    """
    fruit = mask > 0
    total = int(fruit.sum())
    if total == 0:
        return mask, 0.0

    green = fruit & (hue >= 35) & (hue <= 85)
    green_frac = green.sum() / total
    if not (0 < green_frac < green_dominance_thresh):
        return mask, green_frac          # nothing to strip, or green is the fruit itself

    stripped = mask.copy()
    stripped[green] = 0
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    stripped = cv.morphologyEx(stripped, cv.MORPH_OPEN, kernel, iterations=1)
    stripped = cv.morphologyEx(stripped, cv.MORPH_CLOSE, kernel, iterations=1)
    stripped, _ = _select_best_blob(stripped)
    return stripped, green_frac


def _fill_specular_gaps(mask, saturation, value,
                        v_thresh=150, s_thresh=100, max_comp_frac=0.05):
    """
    Patch small BRIGHT + DESATURATED notches bitten out of an otherwise
    fruit-shaped blob -- the signature of a specular highlight (a glare spot
    on shiny fruit skin) that Otsu misreads as background because glare has
    very low saturation, even though it's genuinely part of the fruit surface.

    Safety against filling the WRONG thing (e.g. a gap between two touching
    fruit, or plain background between banana fingers in a bunch):
        - Only components smaller than `max_comp_frac` of the fruit's own
          area are considered -- a highlight is a small local patch; a real
          background gap between fruit lobes is usually much bigger.
        - Only components that are BOTH bright (Value >= v_thresh) AND
          desaturated (Saturation <= s_thresh) qualify -- a leaf/stem notch is
          darker and more saturated (green), so it is correctly left alone.

    Caller is expected to only invoke this when the blob ALREADY looks
    reasonably fruit-shaped (see the circularity gate in _segment_saturation)
    -- this function smooths a good segmentation, it does not try to rescue a
    bad one.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    contour = max(contours, key=cv.contourArea)
    fruit_area = cv.contourArea(contour)

    hull = cv.convexHull(contour)
    hull_mask = np.zeros_like(mask)
    cv.drawContours(hull_mask, [hull], -1, 255, thickness=cv.FILLED)
    gap = (hull_mask > 0) & (mask == 0)
    if not gap.any():
        return mask

    gap_u8 = gap.astype(np.uint8) * 255
    n_labels, labels = cv.connectedComponents(gap_u8)
    filled = mask.copy()
    for label in range(1, n_labels):
        component = labels == label
        size = int(component.sum())
        if size < 5 or size > max_comp_frac * fruit_area:
            continue                                    # too big -> not a highlight
        if value[component].mean() >= v_thresh and saturation[component].mean() <= s_thresh:
            filled[component] = 255
    return filled


def _segment_saturation(img, min_area_frac=0.01, circularity_gate=0.5):
    """
    Saturation + Otsu segmentation, with SHAPE-based blob selection and a
    targeted fix for specular-highlight notches.

    Steps:
        1. Convert BGR -> HSV, take the Saturation channel (fruit is colourful,
           a plain background usually isn't).
        2. Otsu threshold auto-picks the cutoff.
        3. A MILD morphological open+close removes speckle and fills small
           holes without eroding away thin fruit stems.
        4. Instead of blindly keeping the LARGEST connected blob, every blob is
           scored on how fruit-shaped it is (circularity + solidity) and the
           best-scoring one is kept.  This is what stops a big tangle of leaves
           or a background object from winning just because it has more pixels
           than the fruit.
        5. If that blob ALREADY looks reasonably fruit-shaped (circularity >=
           `circularity_gate`), patch any small bright/desaturated notches
           bitten out of it by specular highlights (see _fill_specular_gaps).
           This step is skipped for badly-shaped blobs (e.g. fruit fused with
           leaf/stem) so it never disguises a bad segmentation as a good one.

    Returns (mask, quality) where quality = (circularity, solidity) of the
    FINAL (post-fill) blob.

    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    _, mask = cv.threshold(saturation, 0, 255,
                           cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Mild cleanup only -- aggressive opening shatters the fruit itself
    # (verified experimentally: large kernels erode the fruit before they
    # snap a tight leaf/stem connection).
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

    mask, (circularity, solidity) = _select_best_blob(mask, min_area_frac)

    # Strip a minor leaf/stem attachment BEFORE the highlight fill, so the
    # highlight fill's circularity gate is judged on the leaf-free shape.
    hue = hsv[:, :, 0]
    mask, green_frac = _strip_minor_green(mask, hue)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        _, circularity, solidity, _ = _shape_score(max(contours, key=cv.contourArea))

    if circularity >= circularity_gate:
        mask = _fill_specular_gaps(mask, saturation, value)
        # re-score after the fill so the reported quality reflects the
        # actual final mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            _, circularity, solidity, _ = _shape_score(max(contours, key=cv.contourArea))

    return mask, (circularity, solidity)


def segment_fruit(img, min_area_frac=0.01, return_quality=False):
    """
    Produce a binary mask of the fruit (white = fruit, black = background),
    using Saturation + Otsu thresholding with shape-based blob selection (see
    _segment_saturation).  This assumes nothing about WHERE the fruit sits in
    the frame -- unlike a GrabCut rectangle-init approach, it works whether the
    fruit is centred, off to one side, or only partly in view.

    return_quality: if True, also returns (circularity, solidity) of the
    chosen blob so batch runs can flag low-confidence segmentations for a
    human to check (see batch_preprocess / flag_low_quality).

    Returns: uint8 mask, same H x W as `img`, values in {0, 255}.
    """
    mask, quality = _segment_saturation(img, min_area_frac)
    if return_quality:
        return mask, quality
    return mask


# A blob this jagged/concave is probably fruit merged with leaf/stem/clutter,
# not a clean fruit silhouette -- used to flag images for manual review.
QUALITY_CIRCULARITY_MIN = 0.55
QUALITY_SOLIDITY_MIN = 0.80


def is_low_quality_mask(circularity, solidity):
    """True if the segmented blob doesn't look fruit-shaped enough to trust."""
    return circularity < QUALITY_CIRCULARITY_MIN or solidity < QUALITY_SOLIDITY_MIN


def apply_mask(img, mask):
    """Black-out the background so only the fruit pixels remain."""
    return cv.bitwise_and(img, img, mask=mask)


# ----------------------------------------------------------------------------
# 6. Full pipeline for a single image
# ----------------------------------------------------------------------------
def preprocess_image(path, denoise_method="bilateral", contrast_method="clahe"):
    """
    Run the whole preprocessing chain on one image.

    Returns a dict of every intermediate stage plus a `quality` flag (handy for
    the report figures and for the data-analysis dashboard later):
        original, calibrated, denoised, enhanced, mask, isolated,
        circularity, solidity, low_quality
    """
    original = load_image(path)
    calibrated = resize_keep_aspect(original)
    denoised = denoise(calibrated, denoise_method)
    enhanced = enhance_contrast(denoised, contrast_method)
    mask, (circularity, solidity) = segment_fruit(enhanced, return_quality=True)
    isolated = apply_mask(enhanced, mask)

    return {
        "original": original,
        "calibrated": calibrated,
        "denoised": denoised,
        "enhanced": enhanced,
        "mask": mask,
        "isolated": isolated,
        "circularity": circularity,
        "solidity": solidity,
        "low_quality": is_low_quality_mask(circularity, solidity),
    }


# ----------------------------------------------------------------------------
# 7. Visualisation (before / after figure)
# ----------------------------------------------------------------------------
def visualize_stages(stages, title=None, save_path=None):
    """
    Show every stage in one matplotlib figure, converting BGR -> RGB for
    correct colours (mask is shown in grayscale).
    """
    order = ["original", "calibrated", "denoised", "enhanced", "mask", "isolated"]
    titles = {
        "original": "Original",
        "calibrated": "Calibrated (256x256)",
        "denoised": "Denoised",
        "enhanced": "Contrast Enhanced",
        "mask": "Fruit Mask",
        "isolated": "Isolated Fruit",
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, key in zip(axes.ravel(), order):
        img = stages[key]
        panel_title = titles[key]
        if key == "mask":
            ax.imshow(img, cmap="gray")
            flag = "  [LOW CONFIDENCE]" if stages.get("low_quality") else ""
            panel_title += (f"\ncirc={stages['circularity']:.2f} "
                            f"solid={stages['solidity']:.2f}{flag}")
        else:
            ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.set_title(panel_title, fontsize=10 if key == "mask" else 12)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# 8. Batch processing over the dataset folder
# ----------------------------------------------------------------------------
def batch_preprocess(input_dir, output_dir,
                     denoise_method="bilateral", contrast_method="clahe",
                     save_mask=True):
    """
    Walk `input_dir` recursively, preprocess every image, and mirror the folder
    structure under `output_dir`.  This works regardless of how the Kaggle
    dataset nests its class / ripeness sub-folders, because it simply preserves
    whatever relative path each image already has.

    For each image it writes:
        <output_dir>/<same relative path>/<name>_clean.png   (isolated fruit)
        <output_dir>/<same relative path>/<name>_mask.png    (binary mask)

    Images where the segmented blob doesn't look fruit-shaped (fruit fused
    with leaf/stem/clutter -- see is_low_quality_mask) are NOT dropped, but
    their paths are written to `<output_dir>/_review_needed.txt` so you can
    eyeball just those instead of the whole dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    processed, failed, flagged = 0, 0, []

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(VALID_EXTS):
                continue

            src = os.path.join(root, fname)
            rel = os.path.relpath(root, input_dir)
            dst_dir = os.path.join(output_dir, rel)
            os.makedirs(dst_dir, exist_ok=True)
            stem = os.path.splitext(fname)[0]

            try:
                stages = preprocess_image(src, denoise_method, contrast_method)
                cv.imwrite(os.path.join(dst_dir, f"{stem}_clean.png"),
                           stages["isolated"])
                if save_mask:
                    cv.imwrite(os.path.join(dst_dir, f"{stem}_mask.png"),
                               stages["mask"])
                if stages["low_quality"]:
                    flagged.append(os.path.join(rel, fname))
                processed += 1
            except Exception as exc:                    # noqa: BLE001
                print(f"  [skip] {src}: {exc}")
                failed += 1

    if flagged:
        review_path = os.path.join(output_dir, "_review_needed.txt")
        with open(review_path, "w") as fh:
            fh.write("\n".join(flagged))
        print(f"\n{len(flagged)} image(s) flagged as low-confidence "
              f"segmentation -> {review_path}")

    print(f"Done. Processed {processed} image(s), {failed} failed.")
    return processed, failed


# ----------------------------------------------------------------------------
# 9. Command-line entry point
# ----------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fruit ripeness image preprocessing pipeline.")
    parser.add_argument("input", help="Image file OR dataset folder.")
    parser.add_argument("-o", "--output", default="preprocessed",
                        help="Output folder (batch mode).")
    parser.add_argument("--denoise", default="bilateral",
                        choices=["gaussian", "median", "bilateral", "nlmeans"])
    parser.add_argument("--contrast", default="clahe",
                        choices=["clahe", "stretch"])
    parser.add_argument("--show", action="store_true",
                        help="Show the stage figure (single-image mode).")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        batch_preprocess(args.input, args.output, args.denoise, args.contrast)
    else:
        stages = preprocess_image(args.input, args.denoise, args.contrast)
        fig = visualize_stages(stages, title=os.path.basename(args.input))
        out = os.path.splitext(args.input)[0] + "_stages.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"Saved stage figure -> {out}")
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()