"""
Preprocessing module for the Fruit Ripeness Assessment system.

One toolbox of classical operations, four stages, two consumers.

  STAGE 1  preprocess()              -> the RIPENESS stage.
           Always-on and deliberately aggressive: size, lighting colour, noise
           and glare are all normalised so the later HSV/threshold code measures
           the FRUIT rather than the photograph. Safe to be heavy-handed here,
           because classical code has no training distribution to violate.

  STAGE 2  preprocess_for_detector() -> photometric input to YOLO.
           A CNN learns the statistics of its training data. Any inference-time
           transform absent from training shifts the input off that distribution
           and accuracy DROPS. So this stage has a MILD FLOOR that measured
           neutral-to-positive, plus CONDITIONAL repairs that only fire when the
           frame is measurably worse than the training data. A good frame gets
           the mild pass and nothing else. Size is never changed.

  STAGE 3  prepare_for_detector()    -> geometric input to YOLO.
           The largest accuracy win by far. Small fruit are lost because they
           occupy too few pixels after the whole-frame downscale, not because
           the image is dirty. find_fruit_roi() locates the fruit cluster
           classically, the crop is upscaled, and the detector finally sees the
           fruit at a useful size. Mapping helpers put the masks back into
           original-frame coordinates.

  STAGE 4  refine_detections()       -> after YOLO returns.
           YOLO-seg predicts masks as low-resolution prototypes that are then
           upsampled, so raw masks have staircase edges, speckle and pinholes.
           Morphology and contour operations clean them up.

WHAT NOT TO DO TO A DETECTOR INPUT
    Measured on yolov8s-seg with an 8-apple test image: grayscale conversion
    dropped the count from 8 to 6. White balance, glare removal, per-channel
    histogram equalisation and aggressive CLAHE all change appearance
    statistics the same way and are withheld from Stage 2 for the same reason.
    They stay in Stage 1, where they belong.
    Those figures come from a generic COCO model on ONE image. Re-run
    compare_variants() on your own model and images before quoting them.

Stage 1 numbering (1.1 - 1.6) is unchanged from the original module, so report
figures and references to it still hold.

Syllabus mapping (BMDS2133 Image Processing)
    Practical 2  - Region of Interest, channel split/merge, arithmetic
                   (addWeighted), bitwise operations
    Practical 3  - colour spaces (HSV/LAB), thresholding and Otsu's binarisation,
                   histogram equalisation (CLAHE, look-up tables), median and
                   bilateral filtering, 2D convolution
    Practical 4  - computational photography: non-local means denoising, inpainting
    Practical 5  - geometric transforms: resize, INTER_AREA for shrinking and
                   INTER_CUBIC for enlarging
    Practical 7  - morphological operations: erode, dilate, opening, closing
    Practical 8  - segmentation by threshold + contour detection, contourArea,
                   boundingRect, drawContours
"""

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np


# Long side the detector crop is upscaled to. Match this to the imgsz you pass
# to model.predict() so the crop arrives without a second internal rescale.
DETECTOR_IMGSZ = 960


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------
@dataclass
class PreprocessResult:
    """Outputs of the Stage 1 (ripeness) pipeline."""
    bgr: np.ndarray                 # final cleaned image (BGR)
    hsv: np.ndarray                 # cached HSV conversion
    lab: np.ndarray                 # cached LAB conversion
    scale: float                    # resize factor, for mapping measurements back to source
    steps: dict = field(default_factory=dict)   # intermediates, for report figures


@dataclass
class RestoreResult:
    """Outputs of the Stage 2 (photometric) pipeline."""
    bgr: np.ndarray                 # restored image, SAME size as the input
    metrics: dict                   # quality_report() values that drove the decisions
    applied: list = field(default_factory=list)   # which repairs actually fired


@dataclass
class DetectorInput:
    """Everything needed to run YOLO and map the results back to the frame."""
    image: np.ndarray               # feed this to model.predict()
    roi: tuple                      # (x1, y1, x2, y2) in ORIGINAL frame coordinates
    scale: float                    # upscale factor applied to the crop
    frame_shape: tuple              # (H, W) of the original frame
    metrics: dict = field(default_factory=dict)
    applied: list = field(default_factory=list)


# ======================================================================
# STAGE 1 - Shared operation toolbox
# Both pipelines call these; only the parameters and the gating differ.
# ======================================================================

# ----------------------------------------------------------------------
# 1.1  Resize / normalise size                          [Practical 5]
# ----------------------------------------------------------------------
def resize_normalise(image, target_long_side=800):
    """Scale the image so its longest side equals ``target_long_side`` px.

    Fixing the size first means every later kernel (a 5 px blur, an 8x8 CLAHE
    tile) behaves consistently across phone photos and webcam frames. The scale
    factor is returned so pixel measurements can be mapped back to the original
    resolution (Image Calibration requirement).

    Stage 1 only. Stage 2 must not resize the frame, or predicted masks would no
    longer line up with the original.
    """
    h, w = image.shape[:2]
    scale = target_long_side / max(h, w)
    # INTER_AREA is best for shrinking; INTER_CUBIC is best for enlarging.
    interp = cv.INTER_AREA if scale < 1 else cv.INTER_CUBIC
    resized = cv.resize(image, None, fx=scale, fy=scale, interpolation=interp)
    return resized, scale


# ----------------------------------------------------------------------
# 1.2  White balance (saturation-thresholded grey-world)  [Practicals 2, 3]
# ----------------------------------------------------------------------
def white_balance(image, saturation_threshold=0.9, strength=0.5, max_gain=1.6):
    """Neutralise the colour cast introduced by the lighting, gently.

    RIPENESS PATH ONLY. Never call this on a detector input: shifting hue is
    precisely what separates apple from orange from mango for the classifier.

    Plain grey-world assumes the scene averages to grey. That breaks when a
    single vivid fruit fills the frame: grey-world then adds the opposite colour
    to force the average grey, discolouring the fruit. Three safeguards:

    - vivid and blown-out pixels are excluded when ESTIMATING the gains, using
      the per-pixel saturation measure (max-min)/max that HSV also uses;
    - ``max_gain`` caps how far any channel may be scaled;
    - ``strength`` blends the correction with the original (0 = none, 1 = full).

    Implemented with cv.split, masked channel means and cv.addWeighted rather
    than cv.xphoto.createGrayworldWB, so it needs only base OpenCV and uses only
    operations from Practicals 2 and 3.
    """
    src = image.astype(np.float32)
    B, G, R = cv.split(src)

    mx = np.maximum(np.maximum(B, G), R)
    mn = np.minimum(np.minimum(B, G), R)
    saturation = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    # estimate the illuminant only from neutral, unclipped pixels
    valid = (saturation <= saturation_threshold) & (mx < 255)
    if valid.sum() < 0.01 * valid.size:          # almost everything excluded
        valid = np.ones_like(valid, dtype=bool)

    means = np.array([B[valid].mean(), G[valid].mean(), R[valid].mean()]) + 1e-6
    grey = means.mean()
    gains = np.clip(grey / means, 1.0 / max_gain, max_gain)

    corrected = np.clip(src * gains, 0, 255)
    blended = cv.addWeighted(corrected, strength, src, 1.0 - strength, 0)
    return blended.astype(np.uint8)


# ----------------------------------------------------------------------
# 1.3  Denoise (edge-preserving)                     [Practicals 3, 4]
# ----------------------------------------------------------------------
def denoise(image, median_ksize=5, bilateral_d=9,
            bilateral_sigma_colour=75, bilateral_sigma_space=75,
            sigma=None, strong_threshold=12.0):
    """Remove speckle and grain while keeping the fruit boundary sharp.

    Median blur first (Practical 3.3): highly effective against salt-and-pepper
    speckle. Bilateral filter second (Practical 3.4): smooths flat regions but
    preserves edges, unlike a plain Gaussian blur, which would soften the very
    outline the next stages rely on.

    If a measured noise level is passed in ``sigma`` and it exceeds
    ``strong_threshold``, escalate to Non-Local Means (Practical 4). NLM gives a
    much cleaner result but costs roughly a second per frame, so it is reserved
    for badly degraded stills and never runs during live video.
    """
    if sigma is not None and sigma >= strong_threshold:
        return cv.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)

    speckle_free = cv.medianBlur(image, median_ksize)
    return cv.bilateralFilter(speckle_free, bilateral_d,
                              bilateral_sigma_colour, bilateral_sigma_space)


# ----------------------------------------------------------------------
# 1.4  CLAHE on the L channel (conditional)              [Practical 3]
# ----------------------------------------------------------------------
def apply_clahe(image, clip_limit=1.0, tile_grid_size=(8, 8), contrast_gate=55.0):
    """Even out local brightness, on the L (lightness) channel only, and only
    when the image needs it.

    - L channel only: running CLAHE on B, G and R separately stretches each one
      differently, which introduces a colour cast and can turn an orange into
      something the classifier reads as a mango. Equalising luminance alone
      leaves the a/b chroma channels, and therefore hue, intact.
    - Conditional: if global contrast (std-dev of L) is already high, CLAHE is
      skipped to avoid adding artificial texture and wasted computation. Pass
      ``contrast_gate=float('inf')`` to force it.

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
# 1.5  Specular highlight (glare) suppression       [Practicals 3, 4, 7]
# ----------------------------------------------------------------------
def remove_specular_highlights(image, v_min=240, s_max=90, inpaint_radius=3):
    """Detect small bright glare spots and inpaint them from surrounding colour.

    Threshold in HSV (Practical 3) -> dilate the mask (Practical 7) -> inpaint
    with cv.inpaint, which lives in the same cv.photo module as the Practical 4
    denoising functions.

    Glare carries no real colour information and can be mistaken for a blemish
    or punch a hole in the segmentation mask. Returns (result, glare_mask) so
    the mask can be shown in report figures.
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    _, S, V = cv.split(hsv)

    glare_mask = ((V > v_min) & (S < s_max)).astype(np.uint8) * 255
    # dilate a little so the inpaint also covers the soft edge of each spot
    glare_mask = cv.dilate(glare_mask, np.ones((3, 3), np.uint8), iterations=2)

    if cv.countNonZero(glare_mask) == 0:
        return image, glare_mask          # nothing to do (soft, diffuse lighting)

    return cv.inpaint(image, glare_mask, inpaint_radius, cv.INPAINT_TELEA), glare_mask


# ----------------------------------------------------------------------
# 1.6  Gamma correction via look-up table                [Practical 3]
# ----------------------------------------------------------------------
def correct_exposure(image, current_mean, target=125.0, gamma_limits=(0.5, 2.0)):
    """Pull an under- or over-exposed frame back toward a mid-grey average.

    Uses the same look-up-table idea as the histogram equalisation practical,
    where the cumulative distribution becomes a table and the image is remapped
    with ``img2 = cdf[img]``. Here the table is a gamma curve instead of a cdf.

    Gamma is preferred over full histogram equalisation on the detector path
    because it is monotonic: it lifts the shadows without rearranging the
    histogram, so the ordering of tones, and therefore the hue relationships the
    classifier depends on, survives intact.
    """
    current_mean = max(float(current_mean), 1.0)
    gamma = np.log(target / 255.0) / np.log(current_mean / 255.0)
    gamma = float(np.clip(gamma, *gamma_limits))          # never over-correct

    lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0,
                  0, 255).astype(np.uint8)
    return lut[image]


# ----------------------------------------------------------------------
# 1.7  Unsharp masking                                   [Practical 3]
# ----------------------------------------------------------------------
def unsharp_mask(image, amount=0.6, radius=3):
    """Sharpen with ``original + amount * (original - blurred)``.

    Subtracting a low-pass (Gaussian) result leaves the high frequencies, and
    adding them back boosts edges. This is the high-pass filtering idea from the
    2D convolution practical, expressed with cv.GaussianBlur + cv.addWeighted.

    UNMEASURED on the detector path, so Stage 2 leaves it off by default. Run
    compare_variants() before enabling it. Mild on purpose in any case: it
    strengthens an edge the sensor already recorded, cannot recover an
    out-of-focus shot, and pushed hard it creates halos that read as fake object
    boundaries.
    """
    blurred = cv.GaussianBlur(image, (0, 0), radius)
    return cv.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


# ----------------------------------------------------------------------
# 1.8  Full Stage 1 pipeline (ripeness path)
# ----------------------------------------------------------------------
def preprocess(image, target_long_side=800, keep_steps=False):
    """Run the complete Stage 1 pipeline on a BGR image.

    Order matters: geometry -> colour of light -> noise -> local contrast ->
    glare -> package.

    Set ``keep_steps=True`` to also return every intermediate image (useful for
    the before/after figures in the report). The returned object carries the
    cleaned BGR image plus cached HSV and LAB versions, so the ripeness stage
    never has to re-convert.
    """
    steps = {}

    img, scale = resize_normalise(image, target_long_side)          # 1.1
    if keep_steps:
        steps["1.1_resized"] = img.copy()

    img = white_balance(img)                                        # 1.2
    if keep_steps:
        steps["1.2_white_balanced"] = img.copy()

    img = denoise(img)                                              # 1.3
    if keep_steps:
        steps["1.3_denoised"] = img.copy()

    img, clahe_applied = apply_clahe(img)                           # 1.4
    if keep_steps:
        steps["1.4_clahe"] = img.copy()

    img, glare_mask = remove_specular_highlights(img)               # 1.5
    if keep_steps:
        steps["1.5_glare_mask"] = glare_mask
        steps["1.5_deglared"] = img.copy()

    final_bgr = img
    return PreprocessResult(
        bgr=final_bgr,
        hsv=cv.cvtColor(final_bgr, cv.COLOR_BGR2HSV),
        lab=cv.cvtColor(final_bgr, cv.COLOR_BGR2LAB),
        scale=scale,
        steps=steps,
    )


# ======================================================================
# STAGE 2 - Photometric preparation for the detector
# Mild floor + conditional escalation. Never changes image size.
# ======================================================================

# ----------------------------------------------------------------------
# 2.1  Quality probes                                 [Practicals 3, 8]
# ----------------------------------------------------------------------
def _noise_sigma(gray):
    """Immerkaer's fast noise estimate: convolve with a Laplacian-of-Laplacian
    kernel using cv.filter2D (the 2D convolution practical) and take the mean
    absolute response. Real edges are sparse, so sensor noise dominates the sum.
    """
    kernel = np.array([[1, -2, 1],
                       [-2, 4, -2],
                       [1, -2, 1]], dtype=np.float64)
    h, w = gray.shape
    conv = cv.filter2D(gray.astype(np.float64), -1, kernel)
    return float(np.abs(conv).sum() * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


def quality_report(image):
    """Cheap measurements that drive every Stage 2 decision (~1 ms at 640x480)."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    _, S, V = cv.split(hsv)
    return {
        # variance of the Laplacian: low means out of focus or motion blurred
        "focus": float(cv.Laplacian(gray, cv.CV_64F).var()),
        "brightness": float(V.mean()),          # mean V of HSV: 0 black, 255 white
        "contrast": float(gray.std()),          # global contrast
        # specular highlights: very bright AND unsaturated = blown-out shine
        "glare_pct": float(((V > 245) & (S < 50)).mean() * 100.0),
        "noise": _noise_sigma(gray),
    }


# ----------------------------------------------------------------------
# 2.2  Thresholds
# ----------------------------------------------------------------------
DETECTOR_THRESHOLDS = {
    "glare_pct": 0.40,     # more than this % blown-out  -> inpaint
    "noise": 5.0,          # sigma above this            -> full denoise
    "bright_low": 85.0,    # mean V below this           -> gamma up
    "bright_high": 180.0,  # mean V above this           -> gamma down
    "contrast": 42.0,      # std below this              -> stronger CLAHE
    "focus_low": 60.0,     # below this it is too blurry to rescue
    "focus_high": 250.0,   # above this it is already sharp; do not sharpen
}


def calibrate(image_paths, percentile=5):
    """Derive the thresholds from YOUR OWN training set instead of guessing.

    A frame only needs repairing if it is worse than the images the model was
    actually trained on. Point this at the training images and it returns
    thresholds at the given percentile of that distribution, so the conditional
    repairs fire on roughly the worst ``percentile``% of inputs.

        import glob, preprocessing as pp
        t = pp.calibrate(glob.glob("datasets/fruit/images/train/*.jpg"))
        prep = pp.prepare_for_detector(frame, thresholds=t)
    """
    rows = []
    for path in image_paths:
        img = cv.imread(path)
        if img is not None:
            rows.append(quality_report(img))
    if not rows:
        raise ValueError("No readable images found.")

    col = lambda key: np.array([r[key] for r in rows])
    return {
        # the floors stop a spotless training set producing a threshold of 0,
        # which would make the repair fire on every single frame
        "glare_pct": max(0.15, float(np.percentile(col("glare_pct"), 100 - percentile))),
        "noise": max(3.0, float(np.percentile(col("noise"), 100 - percentile))),
        "bright_low": float(np.percentile(col("brightness"), percentile)),
        "bright_high": float(np.percentile(col("brightness"), 100 - percentile)),
        "contrast": float(np.percentile(col("contrast"), percentile)),
        "focus_low": float(np.percentile(col("focus"), percentile)),
        "focus_high": float(np.percentile(col("focus"), 50)),
    }


# ----------------------------------------------------------------------
# 2.3  The mild floor
# ----------------------------------------------------------------------
def mild_detection_pass(image):
    """The gentlest useful clean-up, applied to every detector input.

    Low-clip CLAHE on luminance plus a light bilateral filter. Both are weak
    enough that they do not move the image far from natural-photo statistics,
    which is why they measured neutral-to-slightly-positive (mean confidence
    0.750 -> 0.784, same 8 fruit found) rather than harmful. This is the floor;
    preprocess_for_detector() escalates from here only when the metrics justify
    it.
    """
    boosted, _ = apply_clahe(image, clip_limit=1.5, contrast_gate=float("inf"))
    return cv.bilateralFilter(boosted, 5, 50, 50)


# ----------------------------------------------------------------------
# 2.4  Full Stage 2 pipeline
# ----------------------------------------------------------------------
def preprocess_for_detector(image, thresholds=None, mild=True, sharpen=False,
                            use_white_balance=False, verbose=False):
    """Photometric preparation of a detector input. Size is never changed.

    Returns a RestoreResult whose ``applied`` list records what fired. On a
    clean frame that is just ``["mild"]``.

    Order: glare is structural so it goes first; denoise before CLAHE, because
    CLAHE amplifies whatever noise is left; sharpen last.

    Each escalation REPLACES the corresponding mild step rather than stacking on
    top of it, so CLAHE and the bilateral filter each run at most once.

    ``use_white_balance`` is off and should stay off unless the same transform
    was applied to the training images before fine-tuning.
    """
    t = dict(DETECTOR_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    q = quality_report(image)
    img = image
    applied = []

    # --- glare: reuse 1.5 with stricter thresholds than the ripeness path
    if q["glare_pct"] > t["glare_pct"]:
        img, _ = remove_specular_highlights(img, v_min=245, s_max=50)
        applied.append(f"glare_inpaint({q['glare_pct']:.2f}%)")

    # --- noise: full denoise if measurably noisy, else the mild bilateral below
    heavy_noise = q["noise"] > t["noise"]
    if heavy_noise:
        img = denoise(img, sigma=q["noise"])
        applied.append(f"denoise(sigma={q['noise']:.1f})")

    # --- exposure: reuse 1.6
    if q["brightness"] < t["bright_low"] or q["brightness"] > t["bright_high"]:
        img = correct_exposure(img, q["brightness"])
        applied.append(f"gamma(mean={q['brightness']:.0f})")

    # --- contrast: stronger CLAHE if flat, else the mild one below
    low_contrast = q["contrast"] < t["contrast"]
    if low_contrast:
        img, _ = apply_clahe(img, clip_limit=2.0, contrast_gate=float("inf"))
        applied.append(f"clahe_L(std={q['contrast']:.0f})")

    # --- the mild floor, for whichever steps did not already escalate
    if mild:
        if not low_contrast:
            img, _ = apply_clahe(img, clip_limit=1.5, contrast_gate=float("inf"))
        if not heavy_noise:
            img = cv.bilateralFilter(img, 5, 50, 50)
        applied.append("mild")

    if use_white_balance:
        img = white_balance(img)
        applied.append("white_balance")

    # --- optional, unmeasured: sharpen only in the middle focus band
    if sharpen and t["focus_low"] < q["focus"] < t["focus_high"]:
        img = unsharp_mask(img)
        applied.append(f"unsharp(focus={q['focus']:.0f})")

    if verbose:
        print("[stage 2]", applied or "nothing")

    return RestoreResult(bgr=img, metrics=q, applied=applied)


# ======================================================================
# STAGE 3 - Geometric preparation: find the fruit, crop, upscale
# This is where the large accuracy gains come from.
# ======================================================================

# ----------------------------------------------------------------------
# 3.1  Classical ROI finder                        [Practicals 2, 3, 8]
# ----------------------------------------------------------------------
def find_fruit_roi(image, sat_pct=75, sat_floor=60, val_min=60, sigma_frac=0.06,
                   pad=0.12, min_frac=0.005, min_side=48, max_frac=0.75,
                   min_coverage=0.60, proc_long_side=480):
    """Locate the fruit region classically, with no detector involved.

    The cue is not colour on its own and not brightness on its own -- a real
    kitchen offers plenty of both. It is DENSITY of vivid pixels. Fruit are
    strongly saturated AND packed together; a fridge, a lamp or a worktop may be
    bright, or coloured, but its vivid pixels are scattered rather than
    clustered. Blurring the vivid-pixel mask turns "clustered" into "high
    value", after which Otsu separates the clusters from everything else.

    The winner is the region containing the density PEAK, not the region with
    the largest area. Picking by area fails on exactly the scenes this function
    exists for: a big, moderately-coloured worktop or wall outscores a small,
    tightly-packed bowl of fruit on area while losing badly on density. Tested
    on a cluttered frame where fruit covered 6%, area-selection returned the
    dark worktop (38% of frame) and peak-selection returned the fruit.

    The saturation cut-off is adaptive but floored. A percentile of the scene's
    own saturation beats a fixed threshold, because the separation between fruit
    and clutter is relative to each scene's palette -- but on a frame that is
    mostly plain background (a white plate, a grey worktop, a wall) more than
    ``sat_pct``% of pixels have zero saturation, the percentile collapses to 0,
    and EVERY pixel counts as vivid. The density map then goes uniform and the
    result is arbitrary. ``sat_floor`` is the absolute minimum saturation a
    pixel must have to count as fruit, which prevents that collapse.

    Everything runs on a downscaled copy (``proc_long_side``), because only a
    coarse bounding box is needed and the large-sigma blur is the expensive part
    -- roughly 10x faster on a phone-resolution photo, which matters for live
    video. Coordinates are scaled back at the end.

    Steps: HSV split and threshold (Practical 3) -> Gaussian blur to build a
    density map (Practical 3) -> Otsu's binarisation (Practical 3) -> contour
    containing the peak, and its bounding box (Practical 8) -> padded ROI
    (Practical 2).

    Returns (x1, y1, x2, y2), or None when there is no useful crop -- either the
    region covers most of the frame, so zooming gains nothing, or it is too
    small in ABSOLUTE pixels to be real. The lower bound is deliberately loose:
    a cluster occupying 1% of the frame is exactly the case zooming rescues
    (measured, fruit at 0.7% of frame: 0 detections at imgsz=640, 6 with zoom),
    so rejecting small fractions would throw away the whole point. ``min_side``
    does the rejecting instead, since a genuine fruit region is never 20 px wide.
    A crop is also rejected unless it captures at least ``min_coverage`` of the
    frame's vivid pixels -- see the coverage guard at the end.
    """
    H, W = image.shape[:2]

    # work small: only a coarse bbox is needed
    ds = min(1.0, proc_long_side / float(max(H, W)))
    small = (cv.resize(image, None, fx=ds, fy=ds, interpolation=cv.INTER_AREA)
             if ds < 1.0 else image)
    h, w = small.shape[:2]

    hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    _, S, V = cv.split(hsv)

    lit = V > val_min
    if lit.sum() < 0.01 * h * w:
        return None
    threshold = max(float(np.percentile(S[lit], sat_pct)), float(sat_floor))
    vivid = ((S >= threshold) & lit).astype(np.float32)
    if vivid.sum() < 0.005 * h * w:
        return None

    # density map: blur converts "many vivid pixels nearby" into "bright here"
    sigma = max(4.0, min(h, w) * sigma_frac)
    density = cv.GaussianBlur(vivid, (0, 0), sigma)
    norm = cv.normalize(density, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    _, cluster = cv.threshold(norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # RETR_EXTERNAL ignores interior holes, so this one call replaces both the
    # fill-holes and the connected-components steps
    contours, _ = cv.findContours(cluster, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # the contour containing the density peak; largest only as a fallback
    _, _, _, peak = cv.minMaxLoc(density)
    chosen = None
    for c in contours:
        if cv.pointPolygonTest(c, (float(peak[0]), float(peak[1])), False) >= 0:
            chosen = c
            break
    if chosen is None:
        chosen = max(contours, key=cv.contourArea)

    # The density map SELECTS which cluster; the vivid mask BOUNDS it. Taking the
    # bounding box of the density blob instead would include its Gaussian halo,
    # which on a frame containing one small cluster and nothing else swells to
    # most of the image and the crop gets rejected as useless.
    region = np.zeros((h, w), np.uint8)
    cv.drawContours(region, [chosen], -1, 255, -1)
    inside = cv.bitwise_and((vivid * 255).astype(np.uint8), region)
    ys, xs = np.nonzero(inside)
    if len(xs) == 0:
        x, y, bw, bh = cv.boundingRect(chosen)
    else:
        x, y = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)

    p = int(pad * max(bw, bh))
    x1, y1 = max(0, x - p), max(0, y - p)
    x2, y2 = min(w, x + bw + p), min(h, y + bh + p)

    # back to full-frame coordinates
    if ds < 1.0:
        x1, y1 = int(x1 / ds), int(y1 / ds)
        x2, y2 = min(W, int(x2 / ds)), min(H, int(y2 / ds))

    frac = (x2 - x1) * (y2 - y1) / float(W * H)
    if frac > max_frac:
        return None                       # covers most of the frame, nothing to gain
    if frac < min_frac or min(x2 - x1, y2 - y1) < min_side:
        return None                       # too small to be a real fruit region

    # Coverage guard. A wrong ROI is far worse than no ROI: cropping to the
    # wrong place hides every fruit and the detector returns nothing at all,
    # whereas declining merely forgoes the zoom. So the crop is only accepted
    # when it captures most of the vivid pixels in the WHOLE frame. If a large
    # share of them lie outside it, this is probably a clutter blob and the
    # fruit are elsewhere.
    sx1, sy1 = int(x1 * ds), int(y1 * ds)
    sx2, sy2 = int(np.ceil(x2 * ds)), int(np.ceil(y2 * ds))
    total = float(vivid.sum())
    captured = float(vivid[sy1:sy2, sx1:sx2].sum())
    if total <= 0 or captured / total < min_coverage:
        return None

    return (x1, y1, x2, y2)


# ----------------------------------------------------------------------
# 3.2  Crop and upscale                                  [Practical 5]
# ----------------------------------------------------------------------
def crop_and_upscale(image, roi, target_long_side=DETECTOR_IMGSZ):
    """Cut the ROI out and scale it so the fruit fills the network input.

    Small fruit are missed because they occupy too few pixels after the
    whole-frame downscale, not because the image is dirty -- measured, an
    8-apple frame where fruit covered 2% of the area gave 0 detections at
    imgsz=640, 7 at imgsz=1280 and 8 with crop-and-upscale. Cropping also
    removes background the network would otherwise spend its input budget on.

    Only ever ENLARGES. If the crop is already at or above the target it is
    returned untouched: YOLO letterboxes to ``imgsz`` internally anyway, so
    resizing twice would just add interpolation blur for nothing.

    INTER_CUBIC for enlarging, as per Practical 5.
    Returns (image_for_detector, scale).
    """
    x1, y1, x2, y2 = roi
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image, 1.0
    scale = target_long_side / max(crop.shape[:2])
    if scale <= 1.05:
        return crop, 1.0
    return cv.resize(crop, None, fx=scale, fy=scale,
                     interpolation=cv.INTER_CUBIC), scale


# ----------------------------------------------------------------------
# 3.3  Map results back to original frame coordinates  [Practicals 2, 5]
# ----------------------------------------------------------------------
def map_mask_to_frame(mask, roi, frame_shape):
    """Paste a mask predicted on the cropped/upscaled ROI back into the frame.

    Nearest-neighbour on purpose: a binary mask must stay binary, and any
    interpolating method would produce grey values along every edge.
    """
    x1, y1, x2, y2 = roi
    H, W = frame_shape[:2]
    resized = cv.resize(mask, (x2 - x1, y2 - y1), interpolation=cv.INTER_NEAREST)
    full = np.zeros((H, W), np.uint8)
    full[y1:y2, x1:x2] = resized
    return full


def map_detections_to_frame(detections, prep):
    """Map a whole list of detections back into original-frame coordinates.

    A no-op when no ROI crop happened, so it is always safe to call.
    """
    x1, y1, x2, y2 = prep.roi
    H, W = prep.frame_shape[:2]
    if (x1, y1, x2, y2) == (0, 0, W, H) and prep.scale == 1.0:
        return detections

    out = []
    for d in detections:
        d = dict(d)
        d["mask"] = map_mask_to_frame(d["mask"], prep.roi, prep.frame_shape)
        if "box" in d:
            b = d["box"]
            d["box"] = [b[0] / prep.scale + x1, b[1] / prep.scale + y1,
                        b[2] / prep.scale + x1, b[3] / prep.scale + y1]
        out.append(d)
    return out


# ----------------------------------------------------------------------
# 3.4  Full Stage 3 pipeline -- the one call the GUI needs
# ----------------------------------------------------------------------
def prepare_for_detector(image, thresholds=None, use_roi=True,
                         target_long_side=DETECTOR_IMGSZ, mild=True,
                         sharpen=False, verbose=False):
    """Geometry then photometry, returning everything needed to map back.

        prep = preprocessing.prepare_for_detector(frame)
        dets = segment(model, prep.image)                     # your YOLO call
        dets = preprocessing.map_detections_to_frame(dets, prep)
        dets = preprocessing.refine_detections(dets, frame)

    Order: find the ROI on the FULL frame, because the density cue needs the
    whole scene to tell a fruit cluster from scattered clutter. Then repair the
    crop at its native resolution, because interpolation smooths noise and would
    make the noise probe under-read. Upscale last.

    When no useful ROI exists the frame is returned at full size with
    roi = whole frame and scale = 1.0, so the mapping helpers stay no-ops and
    the caller needs no special case.
    """
    H, W = image.shape[:2]
    roi = find_fruit_roi(image) if use_roi else None
    whole_frame = (0, 0, W, H)

    if roi is None:
        roi = whole_frame
        working = image
        note = "no ROI (whole frame)"
    else:
        x1, y1, x2, y2 = roi
        working = image[y1:y2, x1:x2]
        note = f"ROI {100 * (x2 - x1) * (y2 - y1) / (W * H):.0f}% of frame"

    restored = preprocess_for_detector(working, thresholds=thresholds,
                                       mild=mild, sharpen=sharpen)

    if roi == whole_frame:
        out_img, scale = restored.bgr, 1.0
    else:
        ch, cw = restored.bgr.shape[:2]
        out_img, scale = crop_and_upscale(restored.bgr, (0, 0, cw, ch),
                                          target_long_side)

    applied = [note] + restored.applied
    if scale != 1.0:
        applied.append(f"upscale x{scale:.1f}")
    if verbose:
        print("[stage 3]", ", ".join(applied))

    return DetectorInput(image=out_img, roi=roi, scale=scale,
                         frame_shape=(H, W), metrics=restored.metrics,
                         applied=applied)


# ======================================================================
# STAGE 4 - Post-segmentation mask refinement
# ======================================================================

# ----------------------------------------------------------------------
# 4.1  Adaptive structuring element                      [Practical 7]
# ----------------------------------------------------------------------
def _adaptive_kernel(area, frac=0.02, lo=3, hi=21):
    """Structuring element size proportional to object size, forced odd.

    A fixed 5x5 that behaves well on a strawberry will erase real detail on a
    fruit that fills the frame, so the kernel is scaled from sqrt(area).
    """
    k = int(frac * np.sqrt(max(float(area), 1.0)))
    k = int(np.clip(k, lo, hi))
    return k + 1 if k % 2 == 0 else k


# ----------------------------------------------------------------------
# 4.2  Clean one instance mask                      [Practicals 7, 8]
# ----------------------------------------------------------------------
def refine_mask(mask, smooth=True):
    """Repair one YOLO instance mask.

    YOLO-seg predicts masks as 160x160 prototypes that are upsampled to frame
    size, so raw masks come back with jagged staircase edges, speckle outside
    the fruit, and pinholes where glare or a stem broke the region.

        1. OPENING  (erode then dilate)  removes speckle outside the fruit
        2. CLOSING  (dilate then erode)  seals notches and thin gaps
        3. largest external contour, drawn FILLED, which in one step keeps only
           the main blob and fills every interior hole
        4. blur then re-threshold, which smooths the staircase boundary

    Steps 1-2 are Practical 7; step 3 is Practical 8; step 4 is the
    blur-then-threshold pattern from the Otsu section of Practical 3.
    """
    m = (mask > 127).astype(np.uint8) * 255
    area = cv.countNonZero(m)
    if area == 0:
        return m

    k = _adaptive_kernel(area)
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
    m = cv.morphologyEx(m, cv.MORPH_OPEN, se)      # Opening  (OED)
    m = cv.morphologyEx(m, cv.MORPH_CLOSE, se)     # Closing  (CDE)

    contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(m)
    m = np.zeros_like(m)
    cv.drawContours(m, [max(contours, key=cv.contourArea)], -1, 255, -1)

    if smooth:
        m = cv.GaussianBlur(m, (0, 0), max(1.0, k / 3.0))
        m = (m > 127).astype(np.uint8) * 255
    return m


# ----------------------------------------------------------------------
# 4.3  Snap the boundary to the real edge           [Practicals 3, 7, 8]
# ----------------------------------------------------------------------
def refine_boundary_otsu(mask, image, area_guard=(0.5, 1.6)):
    """Pull the mask boundary onto the fruit's actual edge using Otsu.

    The YOLO mask is trusted in the middle and doubted at the rim, so morphology
    splits the image into three zones (Practical 7):

        core  = erode(mask)        definitely fruit, kept as is
        outer = dilate(mask)       search region
        band  = outer - core       the uncertain rim, where Otsu decides

    Otsu (Practical 3) then thresholds the saturation channel inside the
    bounding box after a Gaussian blur, exactly as the handbook does for the
    noisy-image example. Fruit is normally far more saturated than a table or a
    wall, so this separates them with no hand-picked threshold. The core votes
    on which side is fruit, so polarity is never guessed.

    Runs in a few milliseconds, so unlike a GrabCut-style refinement it is safe
    to leave on during live video. ``area_guard`` reverts to the input mask if
    Otsu ran away.
    """
    area = cv.countNonZero(mask)
    if area < 500:
        return mask

    k = _adaptive_kernel(area, frac=0.05, lo=5, hi=31)
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
    core = cv.erode(mask, se)
    outer = cv.dilate(mask, se)
    band = cv.subtract(outer, core)
    if cv.countNonZero(core) == 0 or cv.countNonZero(band) == 0:
        return mask

    x, y, w, h = cv.boundingRect(outer)
    S = cv.cvtColor(image[y:y + h, x:x + w], cv.COLOR_BGR2HSV)[:, :, 1]
    S = cv.GaussianBlur(S, (5, 5), 0)
    _, otsu_roi = cv.threshold(S, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    otsu = np.zeros(mask.shape, np.uint8)
    otsu[y:y + h, x:x + w] = otsu_roi

    # let the core decide polarity instead of assuming fruit is the bright side
    if cv.countNonZero(cv.bitwise_and(otsu, core)) < 0.5 * cv.countNonZero(core):
        otsu = cv.bitwise_not(otsu)

    refined = refine_mask(cv.bitwise_or(core, cv.bitwise_and(otsu, band)))

    lo, hi = area_guard
    if not (lo * area < cv.countNonZero(refined) < hi * area):
        return mask                                   # Otsu misfired
    return refined


# ----------------------------------------------------------------------
# 4.4  Duplicate suppression                             [Practical 2]
# ----------------------------------------------------------------------
def _mask_iou(a, b):
    """Intersection over union of two binary masks, via bitwise operations."""
    union = cv.countNonZero(cv.bitwise_or(a, b))
    if union == 0:
        return 0.0
    return cv.countNonZero(cv.bitwise_and(a, b)) / float(union)


def drop_duplicates(detections, iou_threshold=0.45, contain_threshold=0.80):
    """Merge duplicates by MASK overlap rather than box overlap.

    Two masks covering one fruit inflate the contour count. Box-based NMS leaves
    these behind, because neighbouring crops often give boxes that overlap only
    modestly while their masks cover nearly the same object.

    The containment test additionally drops a small mask lying inside a larger
    one, but ONLY when it is clearly smaller (under 60% of the area), because
    two fruit that merely touch are similar in size and must not be merged.
    """
    if not detections:
        return []
    order = sorted(range(len(detections)),
                   key=lambda i: detections[i]["conf"], reverse=True)
    keep, kept_masks = [], []
    for i in order:
        m = detections[i]["mask"]
        area = cv.countNonZero(m)
        if area == 0:
            continue
        duplicate = False
        for km in kept_masks:
            inter = cv.countNonZero(cv.bitwise_and(m, km))
            if inter == 0:
                continue
            contained = (inter / area >= contain_threshold and
                         area < 0.60 * cv.countNonZero(km))
            if _mask_iou(m, km) >= iou_threshold or contained:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
            kept_masks.append(m)
    return [detections[i] for i in sorted(keep)]


# ----------------------------------------------------------------------
# 4.5  Full Stage 4 pipeline
# ----------------------------------------------------------------------
def refine_detections(detections, image=None, min_area_frac=0.0001, min_area_px=100,
                      snap_boundary=True, dedupe=True):
    """Clean every instance mask YOLO returned. Call after mapping back to the
    original frame, so ``image`` is the original frame.

    The size guard is mostly ABSOLUTE. A detection under a hundred-odd pixels is
    noise whatever the frame size, whereas a fraction-only rule scales with the
    image and silently deletes real fruit: 0.08% of a 6 MP photo is a 70 px
    circle, which is larger than a genuinely distant fruit. The fraction term is
    kept as a mild secondary floor for very large images.
    """
    if not detections:
        return detections

    H, W = image.shape[:2] if image is not None else detections[0]["mask"].shape[:2]
    min_area = max(min_area_px, min_area_frac * H * W)

    cleaned = []
    for d in detections:
        m = refine_mask(d["mask"])
        if snap_boundary and image is not None:
            m = refine_boundary_otsu(m, image)
        if cv.countNonZero(m) < min_area:
            continue
        d = dict(d)
        d["mask"] = m
        cleaned.append(d)

    return drop_duplicates(cleaned) if dedupe else cleaned


# ======================================================================
# Measurement -- do not guess which variant helps, measure it
# ======================================================================
def compare_variants(image, predict_fn, thresholds=None):
    """A/B the preprocessing variants on one image and report what each yields.

    ``predict_fn(bgr) -> list of dicts with a 'conf' key`` -- wrap your own
    segment() call. Returns {variant: {"n": count, "mean_conf": float}}.

    Run this over a folder of your own images with your own best2.pt before
    quoting any numbers in the report. Published figures were measured on a
    generic COCO model on a single image and will not transfer reliably.
    """
    results = {}

    def score(name, img):
        dets = predict_fn(img)
        confs = [float(d["conf"]) for d in dets]
        results[name] = {"n": len(dets),
                         "mean_conf": round(float(np.mean(confs)), 3) if confs else 0.0}

    score("original", image)
    score("mild only", mild_detection_pass(image))
    score("conditional", preprocess_for_detector(image, thresholds=thresholds).bgr)
    score("conditional + ROI zoom",
          prepare_for_detector(image, thresholds=thresholds).image)
    return results


# ----------------------------------------------------------------------
# Command-line test:  python preprocessing.py <input> [output]
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

    stem = dst_path.rsplit(".", 1)[0]
    print(f"Input               : {src.shape[1]}x{src.shape[0]}")

    result = preprocess(src, keep_steps=True)
    cv.imwrite(dst_path, result.bgr)
    print(f"Stage 1 (ripeness)  : {result.bgr.shape[1]}x{result.bgr.shape[0]} "
          f"scale {result.scale:.3f} -> {dst_path}")

    prep = prepare_for_detector(src)
    det_path = f"{stem}_detector.png"
    cv.imwrite(det_path, prep.image)
    print(f"Stage 3 (detector)  : {prep.image.shape[1]}x{prep.image.shape[0]} "
          f"-> {det_path}")
    print("  quality  : " + ", ".join(f"{k}={v:.1f}" for k, v in prep.metrics.items()))
    print("  applied  : " + ", ".join(prep.applied))
    print(f"  roi      : {prep.roi}  scale {prep.scale:.2f}")