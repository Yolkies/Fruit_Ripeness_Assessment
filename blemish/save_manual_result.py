# pip install opencv-python numpy

"""
Darkest-cluster blemish segmentation over the whole blemish/data set.

Pipeline per image:

    1. fruit_mask()   isolate the fruit from the background
    2. k-means (k=3)  on the FRUIT PIXELS ONLY
    3. darkest cluster -> candidate blemish pixels
    4. area filter, as a FRACTION of fruit area
    5. fill survivors into a binary defect mask

Run from the project root:
    python blemish/save_manual_result.py

Why it is built this way
------------------------
The earlier version was tuned to a single image (data/images.jpg) and broke
on the rest of the set. Three fixes:

* k-means now runs on fruit pixels only. Run over the whole photo, the
  background owns one of the three clusters, so on a dark background the
  "darkest cluster" IS the background - images (1).jpg reported 22.7% defect
  that was entirely the black backdrop.

* The fruit mask no longer assumes the fruit is darker than the background.
  THRESH_BINARY_INV + Otsu on greyscale only works for a light backdrop; it
  selected the hand in the Depositphotos shot and the whole frame for the
  black-backdrop apple. Saturation-Otsu plus a value floor handles light,
  dark and grey backdrops alike.

* Area bounds are fractions of fruit area, not absolute pixels. The set
  ranges from 213x160 to 2000x1333, so a fixed "40..10000 px" window means
  completely different things per image.

Known limit: the fruit must be the dominant saturated object. That holds for
the four studio shots. It does NOT hold for Depositphotos_324874664 (fruit
held in a hand against green foliage) - see the note printed for it. That
image needs the project's YOLO segmentation, not thresholding.
"""

import os

import cv2 as cv
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_ROOT = os.path.join(os.path.dirname(__file__), "results")

CLUSTER = 3

# Area filter, as a fraction of fruit area (was 40 and 10000 absolute px).
# 0.0002 / 0.06 reproduce the original bounds on data/images.jpg.
MIN_AREA_FRAC = 0.0002
MAX_AREA_FRAC = 0.06

# Background detection: how far (CIELAB distance) a pixel may sit from the
# median border colour and still count as backdrop. Low enough to keep fruit
# that happens to be pale, high enough to absorb gradients in a studio sweep.
BG_TOLERANCE = 38

# Shrink the fruit mask before analysis so the boundary rim, which is dark
# against any backdrop, is not clustered as blemish.
BOUNDARY_EROSION_FRAC = 0.02

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def fruit_mask(bgr):
    """Isolate the fruit. Returns a 0/255 mask, or None if nothing found.

    Works by identifying the BACKGROUND rather than the fruit: sample the
    image border, flood-fill everything within BG_TOLERANCE of that colour
    that is connected to an edge, and keep the largest remaining blob.

    This beats thresholding the fruit directly, which has to guess whether
    the fruit is lighter or darker than the backdrop. Measured on this set,
    saturation-Otsu fragmented the black-backdrop apple (its grey scabs are
    unsaturated, so they punched holes that broke the silhouette apart);
    border flood-fill recovers all four studio shots exactly.
    """
    h, w = bgr.shape[:2]
    lab = cv.cvtColor(cv.GaussianBlur(bgr, (5, 5), 0),
                      cv.COLOR_BGR2LAB).astype(np.int16)

    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    median = np.median(border, axis=0)
    distance = np.sqrt(((lab - median) ** 2).sum(axis=2))
    background = (distance < BG_TOLERANCE).astype(np.uint8) * 255

    # Only background touching an edge counts; a dark blemish in the middle
    # of the fruit may match the backdrop colour but is not background.
    connected = np.zeros((h, w), np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
    for seed in seeds:
        if background[seed[1], seed[0]] == 255:
            tmp = background.copy()
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv.floodFill(tmp, ff_mask, seed, 128)
            connected[tmp == 128] = 255

    mask = cv.bitwise_not(connected)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None

    biggest = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    out = np.zeros((h, w), np.uint8)
    out[labels == biggest] = 255

    # Fill interior holes: a dark blemish must stay part of the fruit.
    contours, _ = cv.findContours(out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    out = np.zeros((h, w), np.uint8)
    cv.drawContours(out, contours, -1, 255, -1)
    return out


def erode_boundary(mask):
    """Pull the mask in slightly, dropping the dark rim at the fruit edge."""
    h, w = mask.shape[:2]
    r = int(min(h, w) * BOUNDARY_EROSION_FRAC)
    if r < 1:
        return mask
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv.erode(mask, k, iterations=1)


def darkest_cluster(gray, mask):
    """k-means over fruit pixels only; return (candidate mask, cluster levels).

    Restricting the fit to the fruit is what stops the background from
    claiming a cluster and being reported as the darkest one.
    """
    inside = mask > 0
    vec = np.float32(gray[inside].reshape(-1, 1))
    k = min(CLUSTER, len(np.unique(vec)))
    if k < 2:
        return np.zeros(gray.shape, np.uint8), []

    _c, labels, centers = cv.kmeans(
        vec, k, None,
        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers).ravel()
    quant = np.zeros(gray.shape, np.uint8)
    quant[inside] = centers[labels.flatten()]

    levels = sorted(centers.tolist())
    candidate = ((quant <= levels[0]) & inside).astype(np.uint8) * 255
    return candidate, levels


def analyse(bgr):
    """Full per-image pipeline. Returns a dict of masks + measurements."""
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    fm = fruit_mask(bgr)
    if fm is None:
        return None
    fm = erode_boundary(fm)
    fruit_px = int(cv.countNonZero(fm))
    if fruit_px == 0:
        return None

    candidate, levels = darkest_cluster(gray, fm)

    lo = max(1, int(fruit_px * MIN_AREA_FRAC))
    hi = int(fruit_px * MAX_AREA_FRAC)

    contours, _ = cv.findContours(candidate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    defect = np.zeros(gray.shape, np.uint8)
    kept = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < lo or area > hi:
            continue
        kept += 1
        cv.drawContours(defect, [cnt], -1, 255, -1)

    defect = cv.bitwise_and(defect, fm)          # never count outside the fruit
    defect_px = int(cv.countNonZero(defect))

    overlay = bgr.copy()
    tint = overlay.copy()
    tint[defect > 0] = (0, 0, 255)
    overlay = cv.addWeighted(tint, 0.45, overlay, 0.55, 0)
    cv.drawContours(overlay, cv.findContours(
        defect, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 255), 1)

    return {
        "fruit_mask": fm, "candidate": candidate, "defect": defect,
        "overlay": overlay, "levels": levels, "regions": kept,
        "fruit_px": fruit_px, "defect_px": defect_px,
        "defect_pct": 100.0 * defect_px / fruit_px,
        "area_bounds": (lo, hi),
    }


def main():
    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise SystemExit(f"No images in {DATA_DIR}")

    summary = []
    for name in files:
        path = os.path.join(DATA_DIR, name)
        bgr = cv.imread(path)
        if bgr is None:
            print(f"SKIP (unreadable): {name}")
            continue

        r = analyse(bgr)
        if r is None:
            print(f"SKIP (no fruit found): {name}")
            continue

        stem = os.path.splitext(name)[0]
        out_dir = os.path.join(OUT_ROOT, stem)
        os.makedirs(out_dir, exist_ok=True)
        cv.imwrite(os.path.join(out_dir, "01_fruit_mask.png"), r["fruit_mask"])
        cv.imwrite(os.path.join(out_dir, "02_darkest_cluster.png"), r["candidate"])
        cv.imwrite(os.path.join(out_dir, "03_defect_mask.png"), r["defect"])
        cv.imwrite(os.path.join(out_dir, "04_overlay.png"), r["overlay"])

        lo, hi = r["area_bounds"]
        lines = [
            "DARKEST-CLUSTER BLEMISH SEGMENTATION",
            "=" * 44,
            f"image        : {name}  ({bgr.shape[1]}x{bgr.shape[0]})",
            f"cluster levels: {r['levels']}   (darkest kept: {r['levels'][0]})",
            f"area filter  : {lo}..{hi} px "
            f"({MIN_AREA_FRAC:.2%}..{MAX_AREA_FRAC:.2%} of fruit)",
            "",
            f"regions      : {r['regions']}",
            f"fruit pixels : {r['fruit_px']}",
            f"defect pixels: {r['defect_px']}",
            f"defect       : {r['defect_pct']:.2f}%",
        ]
        with open(os.path.join(out_dir, "result.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        summary.append((name, r["fruit_px"], r["regions"], r["defect_pct"]))

    print(f"\n{'image':46}{'fruit px':>10}{'regions':>9}{'defect':>9}")
    print("-" * 74)
    for name, fp, n, pct in summary:
        print(f"{name[:44]:46}{fp:10d}{n:9d}{pct:8.2f}%")
    print(f"\nsaved -> {os.path.abspath(OUT_ROOT)}")


if __name__ == "__main__":
    main()
