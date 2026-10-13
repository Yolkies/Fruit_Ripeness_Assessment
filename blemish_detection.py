from dataclasses import dataclass

import cv2
import numpy as np

from detection import (
    CLASS_NAMES,
    annotate,
    crop_detection_from_original,
    run_yolo_segmentation,
)


@dataclass
class BlemishParams:
    clusters: int = 3

    min_area_frac: float = 0.0002
    max_area_frac: float = 0.25

    boundary_erosion_frac: float = 0.02

    max_chroma_ratio: float = 0.70

    defect_threshold_pct: float = 5.0

    use_chroma_test: bool = True

    random_seed: int = 42


PARAMS = BlemishParams()


def _erode_boundary(mask, frac):
    h, w = mask.shape[:2]
    r = int(min(h, w) * frac)
    if r < 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask, k, iterations=1)


def _normalise_mask(mask, shape):
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def _chroma(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)


def _darkest_cluster(gray, fruit_mask, k, seed=None):
    inside = fruit_mask > 0
    values = gray[inside]
    if values.size == 0:
        return np.zeros(gray.shape, np.uint8), []

    k = int(min(k, len(np.unique(values))))
    if k < 2:
        return np.zeros(gray.shape, np.uint8), [int(values.min())]

    if seed is not None:
        cv2.setRNGSeed(int(seed))

    _compact, labels, centers = cv2.kmeans(
        np.float32(values.reshape(-1, 1)),
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = np.uint8(centers).ravel()
    quant = np.zeros(gray.shape, np.uint8)
    quant[inside] = centers[labels.flatten()]

    levels = sorted(int(c) for c in centers)
    candidate = ((quant <= levels[0]) & inside).astype(np.uint8) * 255
    return candidate, levels


def assess_single_fruit(original_crop, fruit_mask, params=None):
    p = params or PARAMS

    if original_crop is None or original_crop.size == 0:
        raise ValueError("Blemish stage received an empty crop.")

    h, w = original_crop.shape[:2]
    mask = _normalise_mask(fruit_mask, (h, w))
    if mask is None:
        mask = (np.any(original_crop != 0, axis=2).astype(np.uint8)) * 255

    mask = _erode_boundary(mask, p.boundary_erosion_frac)
    fruit_px = int(cv2.countNonZero(mask))
    if fruit_px == 0:
        raise ValueError("Blemish stage received an empty fruit mask.")

    gray = cv2.cvtColor(original_crop, cv2.COLOR_BGR2GRAY)
    candidate, levels = _darkest_cluster(
        gray, mask, p.clusters, getattr(p, "random_seed", None)
    )

    healthy = (mask > 0) & (candidate == 0)
    chroma = _chroma(original_crop)
    healthy_chroma = float(np.mean(chroma[healthy])) if np.any(healthy) else 0.0

    lo = max(1, int(fruit_px * p.min_area_frac))
    hi = max(lo, int(fruit_px * p.max_area_frac))

    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defect = np.zeros(gray.shape, np.uint8)
    rejected = np.zeros(gray.shape, np.uint8)
    blobs = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < lo or area > hi:
            continue

        blob = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(blob, [cnt], -1, 255, -1)
        sel = blob > 0
        if not np.any(sel):
            continue

        ratio = (
            float(np.mean(chroma[sel])) / healthy_chroma
            if healthy_chroma > 1e-6
            else 0.0
        )

        if p.use_chroma_test and ratio > p.max_chroma_ratio:
            rejected[sel] = 255
            continue

        defect[sel] = 255
        blobs.append(
            {"area_px": int(np.count_nonzero(sel)), "chroma_ratio": round(ratio, 3)}
        )

    defect = cv2.bitwise_and(defect, mask)
    defect_px = int(cv2.countNonZero(defect))
    defect_pct = 100.0 * defect_px / fruit_px
    status = "Defected" if defect_pct >= p.defect_threshold_pct else "Good"

    comparison = ">=" if status == "Defected" else "<"
    result = {
        "status": status,
        "defect_area_pct": round(defect_pct, 2),
        "defect_pixels": defect_px,
        "fruit_pixels": fruit_px,
        "blob_count": len(blobs),
        "largest_blob_px": max((b["area_px"] for b in blobs), default=0),
        "cluster_levels": levels,
        "healthy_chroma": round(healthy_chroma, 1),
        "blobs": blobs,
        "area_bounds_px": [lo, hi],
        "min_area_frac": p.min_area_frac,
        "max_area_frac": p.max_area_frac,
        "defect_threshold_pct": p.defect_threshold_pct,
        "chroma_test": p.use_chroma_test,
        "max_chroma_ratio": p.max_chroma_ratio,
        "rejected_px": int(cv2.countNonZero(rejected)),
        "rule_basis": (
            f"defect area {defect_pct:.2f}% {comparison} threshold "
            f"{p.defect_threshold_pct:.2f}% ({defect_px} of {fruit_px} fruit "
            f"pixels, {len(blobs)} region(s), darkest cluster {levels[0] if levels else '-'})"
        ),
    }

    diagnostics = {
        "01_crop": original_crop,
        "02_fruit_mask": mask,
        "03_darkest_cluster": candidate,
        "04_rejected_as_shadow": rejected,
        "05_defect_mask": defect,
        "06_overlay": overlay_defects(original_crop, defect),
    }
    return result, diagnostics


def format_report(result):
    if "error" in result:
        return (
            f"BLEMISH DETECTION\n{'=' * 52}\n"
            f"fruit         : {result.get('number')} "
            f"({result.get('fruit_type')})\n"
            f"ERROR         : {result['error']}\n"
        )

    levels = result.get("cluster_levels") or []
    lo, hi = result.get("area_bounds_px", [0, 0])
    fruit_px = result.get("fruit_pixels", 0)

    name = str(result.get("fruit_type", "")).title()
    if result.get("fruit_type_overridden"):
        name += f"  (corrected from {str(result.get('yolo_fruit_type')).title()})"

    lines = [
        "DARKEST-CLUSTER BLEMISH SEGMENTATION",
        "=" * 52,
        f"fruit          : {result.get('number')}  {name}",
    ]
    if result.get("confidence") is not None:
        lines.append(f"detection conf : {result['confidence']:.3f}")

    lines += [
        f"cluster levels : {levels}"
        + (f"   (darkest kept: {levels[0]})" if levels else ""),
        f"area filter    : {lo}..{hi} px "
        f"({result.get('min_area_frac', 0):.2%}.."
        f"{result.get('max_area_frac', 0):.2%} of fruit)",
        "shadow test    : "
        + (
            f"on  (reject blob if chroma > "
            f"{result.get('max_chroma_ratio', 0):.2f} of healthy)"
            if result.get("chroma_test")
            else "off"
        ),
        "",
        f"regions        : {result.get('blob_count', 0)}",
        f"fruit pixels   : {fruit_px}",
        f"defect pixels  : {result.get('defect_pixels', 0)}",
        f"largest region : {result.get('largest_blob_px', 0)} px",
        f"rejected shadow: {result.get('rejected_px', 0)} px",
        f"healthy chroma : {result.get('healthy_chroma', 0)}",
        "",
        f"defect         : {result.get('defect_area_pct', 0):.2f}%",
        f"threshold      : {result.get('defect_threshold_pct', 0):.2f}%",
        f"STATUS         : {result.get('status', '-')}",
    ]

    blobs = result.get("blobs") or []
    if blobs:
        lines += [
            "",
            "regions (largest first)",
            "-" * 52,
            f"{'#':>3}  {'area px':>9}  {'% of fruit':>11}  {'chroma':>7}",
        ]
        ordered = sorted(blobs, key=lambda b: -b["area_px"])
        for i, blob in enumerate(ordered, start=1):
            share = 100.0 * blob["area_px"] / fruit_px if fruit_px else 0.0
            lines.append(
                f"{i:>3}  {blob['area_px']:>9}  {share:>10.2f}%"
                f"  {blob['chroma_ratio']:>7.3f}"
            )

    return "\n".join(lines) + "\n"


def overlay_defects(bgr, defect_mask):
    out = bgr.copy()
    if cv2.countNonZero(defect_mask) == 0:
        return out
    tint = out.copy()
    tint[defect_mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)
    contours, _ = cv2.findContours(
        defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (0, 255, 255), 1)
    return out


def assess_detected_fruit(detected_fruit, selected_fruit_type=None, params=None):
    result, diagnostics = assess_single_fruit(
        detected_fruit["crop"], detected_fruit["fruit_mask"], params
    )

    yolo_fruit_type = detected_fruit.get("yolo_fruit_type")
    if selected_fruit_type is None:
        fruit_type = yolo_fruit_type
    else:
        fruit_type = str(selected_fruit_type).lower().strip()

    result["number"] = detected_fruit.get("number")
    result["fruit_type"] = fruit_type
    result["yolo_fruit_type"] = yolo_fruit_type
    result["selected_fruit_type"] = fruit_type
    result["fruit_type_overridden"] = (
        yolo_fruit_type is not None and fruit_type != yolo_fruit_type
    )
    result["confidence"] = detected_fruit.get("confidence")
    result["crop"] = detected_fruit["crop"]
    result["masked_crop"] = detected_fruit.get("masked_crop")
    result["diagnostics"] = diagnostics
    return result


def analyze_image(image_bgr, params=None):
    if image_bgr is None:
        raise ValueError("Input image is empty.")

    detections = run_yolo_segmentation(image_bgr)
    if not detections:
        return {"count": 0, "annotated_image": image_bgr, "fruits": []}

    annotated_image = annotate(image_bgr, detections, show_mask=True)

    fruits = []
    for i, detection in enumerate(detections, start=1):
        original_crop, fruit_mask, background_removed = crop_detection_from_original(
            image_bgr, detection
        )

        try:
            result, diagnostics = assess_single_fruit(original_crop, fruit_mask, params)
        except ValueError as exc:
            fruits.append(
                {
                    "number": i,
                    "fruit_type": CLASS_NAMES[detection["cls"]],
                    "error": str(exc),
                }
            )
            continue

        result["number"] = i
        result["fruit_type"] = CLASS_NAMES[detection["cls"]]
        result["confidence"] = detection["conf"]
        result["crop"] = original_crop
        result["masked_crop"] = background_removed
        result["diagnostics"] = diagnostics
        fruits.append(result)

    defected = sum(1 for f in fruits if f.get("status") == "Defected")
    return {
        "count": len(fruits),
        "defected": defected,
        "good": len(fruits) - defected,
        "annotated_image": annotated_image,
        "fruits": fruits,
    }
