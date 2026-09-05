"""Module 2.2 - Fruit Detection and Instance Segmentation."""

import os
import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import preprocessing


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = str(PROJECT_ROOT / "best4.pt")

CLASS_NAMES = ["apple", "banana", "mango", "orange", "strawberry", "tomato"]
OUTPUT_DIR = "pipeline_output"          # masked crops for the ripeness stage go here
CONF_THRESHOLD = 0.2              # detection confidence cut-off


# ----------------------------------------------------------------------
# Tunable parameters (edited live by the GUI)
# ----------------------------------------------------------------------
@dataclass
class DetectionParams:
    model_path: str = MODEL_PATH
    conf_threshold: float = CONF_THRESHOLD
    mask_binary_threshold: float = 0.5
    dedupe_overlap_threshold: float = 0.4
    retina_masks: bool = True
    overlay_alpha: float = 0.45
    contour_thickness: int = 2
    label_font_scale: float = 0.5
    output_dir: str = OUTPUT_DIR


PARAMS = DetectionParams()


@lru_cache(maxsize=4)
def _load_model_cached(path):
    return YOLO(path)


def load_model():
    """
    Load YOLO only once per model path instead of reloading it every time
    Streamlit reruns the script.
    """
    return _load_model_cached(PARAMS.model_path)


def segment(model, crops, crop_rects, orig_shape, conf=None, params=None):
    """
    Flowchart: Fruit Segmentation.
    Returns a list of per-instance dicts: binary mask, class id, confidence.
    """
    p = params if params is not None else PARAMS
    conf = p.conf_threshold if conf is None else conf

    H, W = orig_shape[:2]
    detections = []

    for crop, (cx, cy, cw, ch) in zip(crops, crop_rects):
        res = model.predict(crop, conf=conf, retina_masks=p.retina_masks, verbose=False)[0]
        if res.masks is None:
            continue

        masks = res.masks.data.cpu().numpy()          # (N, ch, cw) in 0..1
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        for m, c, cf in zip(masks, cls_ids, confs):
            mask_resized = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > p.mask_binary_threshold).astype("uint8") * 255

            global_mask = np.zeros((H, W), dtype="uint8")
            global_mask[cy:cy+ch, cx:cx+cw] = mask_binary

            detections.append({"mask": global_mask, "cls": int(c), "conf": float(cf)})

    return dedupe(detections, p.dedupe_overlap_threshold)


def dedupe(detections, overlap_thr=None):
    overlap_thr = PARAMS.dedupe_overlap_threshold if overlap_thr is None else overlap_thr
    kept = []
    for d in sorted(detections, key=lambda x: -x["conf"]):
        area = cv2.countNonZero(d["mask"])
        if area == 0:
            continue
        if any(cv2.countNonZero(cv2.bitwise_and(d["mask"], k["mask"])) /
               min(area, cv2.countNonZero(k["mask"])) > overlap_thr for k in kept):
            continue
        kept.append(d)
    return sorted(kept, key=lambda d: cv2.boundingRect(d["mask"])[0])


def contour_count(detections):
    """
    Flowchart: Contour Count.
    Classical step: run cv2.findContours on EACH instance mask. Because the
    model separates touching fruit into individual masks, each mask yields one
    clean external contour -> an accurate count even when fruit touch (which
    plain thresholding + contours would merge into one).
    Adds 'contour' and 'area' to each detection; returns the total count.
    """
    count = 0
    for d in detections:
        cnts, _ = cv2.findContours(d["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            d["contour"] = None
            d["area"] = 0
            continue
        largest = max(cnts, key=cv2.contourArea)
        d["contour"] = largest
        d["area"] = float(cv2.contourArea(largest))
        count += 1
    return count


def annotate(img_bgr, detections, show_mask=False, params=None):
    """Draw contours + class labels + index for the Segmented view.
    If show_mask=True, also blend each instance's filled mask as a translucent
    coloured overlay."""
    p = params if params is not None else PARAMS
    alpha = p.overlay_alpha
    thickness = max(1, int(p.contour_thickness))

    out = img_bgr.copy()
    palette = [(66, 66, 244), (48, 172, 220), (60, 76, 231), (80, 175, 76),
               (0, 214, 255), (170, 66, 244)]
    if show_mask:
        overlay = out.copy()
        for d in detections:
            if d.get("contour") is None:
                continue
            color = palette[d["cls"] % len(palette)]
            overlay[d["mask"] > 0] = color
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    for i, d in enumerate(detections):
        if d.get("contour") is None:
            continue
        color = palette[d["cls"] % len(palette)]
        cv2.drawContours(out, [d["contour"]], -1, color, thickness)
        x, y, w, h = cv2.boundingRect(d["contour"])
        label = f"{i+1}:{CLASS_NAMES[d['cls']]} {d['conf']:.2f}"
        cv2.putText(out, label, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, p.label_font_scale, color, thickness, cv2.LINE_AA)
    return out


def run_yolo_segmentation(image_bgr, model=None):
    """
    Module 2.1 preprocessing + YOLO instance segmentation + contour count.
    """
    if model is None:
        model = load_model()

    prep = preprocessing.preprocess(image_bgr)

    if not prep.crops:
        crops = [prep.bgr]
        crop_rects = [(0, 0, prep.bgr.shape[1], prep.bgr.shape[0])]
    else:
        crops = prep.crops
        crop_rects = prep.crop_rects

    detections = segment(
        model,
        crops,
        crop_rects,
        image_bgr.shape,
    )

    contour_count(detections)

    return detections


def crop_detection_from_original(image_bgr, detection):
    """
    Returns:
        original_crop
        mask_crop
        background_removed
    """
    mask_full = detection["mask"]

    if detection.get("contour") is not None:
        x, y, w, h = cv2.boundingRect(detection["contour"])
    else:
        x, y, w, h = cv2.boundingRect(mask_full)

    original_crop = image_bgr[y:y + h, x:x + w].copy()
    mask_crop = mask_full[y:y + h, x:x + w].copy()

    background_removed = cv2.bitwise_and(
        original_crop,
        original_crop,
        mask=mask_crop,
    )

    return original_crop, mask_crop, background_removed


def detect_image(image_bgr):
    """
    Preprocessing + YOLO fruit detection only.
    Ripeness analysis (Module 2.3) is NOT performed here.
    """
    if image_bgr is None:
        raise ValueError("Input image is empty.")

    detections = run_yolo_segmentation(image_bgr)

    if not detections:
        return {
            "count": 0,
            "annotated_image": image_bgr,
            "detections": [],
        }

    annotated_image = annotate(image_bgr, detections, show_mask=True)

    prepared_detections = []

    for i, detection in enumerate(detections, start=1):
        yolo_fruit_type = CLASS_NAMES[int(detection["cls"])]

        (
            original_crop,
            fruit_mask,
            background_removed,
        ) = crop_detection_from_original(image_bgr, detection)

        prepared_detections.append({
            "number": i,
            "yolo_fruit_type": yolo_fruit_type,
            "confidence": float(detection["conf"]),
            "crop": original_crop,
            "fruit_mask": fruit_mask,
            "masked_crop": background_removed,
        })

    return {
        "count": len(prepared_detections),
        "annotated_image": annotated_image,
        "detections": prepared_detections,
    }


def export_for_ripeness(img_bgr, detections, out_dir=None):
    """
    Flowchart hand-off: Output -> (Ripeness / Defect / Grade).
    For each fruit, save a MASKED crop (background removed, only fruit pixels)
    plus a manifest. This is exactly what the HSV ripeness + defect stage
    consumes. Returns the manifest list.
    """
    out_dir = PARAMS.output_dir if out_dir is None else out_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = os.path.join(out_dir, stamp)
    os.makedirs(session, exist_ok=True)
    manifest = []
    for i, d in enumerate(detections):
        if d.get("contour") is None:
            continue
        # isolate this fruit's pixels only (background -> black)
        masked = cv2.bitwise_and(img_bgr, img_bgr, mask=d["mask"])
        x, y, w, h = cv2.boundingRect(d["contour"])
        crop = masked[y:y + h, x:x + w]
        fname = f"fruit_{i+1:02d}_{CLASS_NAMES[d['cls']]}.png"
        cv2.imwrite(os.path.join(session, fname), crop)
        manifest.append({
            "index": i + 1,
            "fruit_type": CLASS_NAMES[d["cls"]],
            "confidence": round(d["conf"], 4),
            "contour_area_px": round(d["area"], 1),
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "masked_crop": fname,
        })
    with open(os.path.join(session, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return session, manifest
