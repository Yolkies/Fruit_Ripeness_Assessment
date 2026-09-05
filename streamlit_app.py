import io
import os
import subprocess
import sys
import zipfile

import cv2
import numpy as np
import streamlit as st

import detection
import preprocessing
import ripeness
import training
from detection import annotate, contour_count, export_for_ripeness, segment
from params_spec import (
    detection_sections,
    preprocessing_sections,
    ripeness_sections,
)

st.set_page_config(page_title="Fruit Ripeness Assessment", layout="wide")


# ============================================================
# RIPENESS RESULT UI CONFIGURATION
# ============================================================

MASK_DISPLAY_CONFIG = {
    "apple": [
        ("red_mask", "Red Area", (0, 0, 255)),
        ("yellow_mask", "Yellow Area", (0, 255, 255)),
        ("other_mask", "Other Area", (180, 180, 180)),
    ],

    "banana": [
        ("yellow_mask", "Yellow Area", (0, 255, 255)),
        ("brown_mask", "Brown Area", (42, 42, 165)),
        ("developed_peel_mask", "Developed Peel Area", (0, 165, 255)),
        ("other_mask", "Other Area", (180, 180, 180)),
    ],

    "mango": [
        ("yellow_mask", "Yellow Area", (0, 255, 255)),
        ("other_mask", "Other Area", (180, 180, 180)),
    ],

    "orange": [
        (
            "yellow_orange_mask",
            "Yellow / Orange Area",
            (0, 165, 255)
        ),
        (
            "non_yellow_orange_mask",
            "Non Yellow / Orange Area",
            (180, 180, 180)
        ),
    ],

    "strawberry": [
        ("red_mask", "Red Area", (0, 0, 255)),
        ("dark_red_mask", "Dark Red Area", (0, 0, 120)),
    ],

    "tomato": [
        ("green_mask", "Green Area", (0, 255, 0)),
        ("turning_mask", "Turning Area", (0, 255, 255)),
        ("pink_mask", "Pink Area", (180, 105, 255)),
        ("mature_red_mask", "Mature Red Area", (0, 0, 255)),
        ("colour_break_mask", "Colour Break Area", (0, 200, 255)),
        ("red_orange_mask", "Red / Orange Area", (0, 100, 255)),
        ("other_mask", "Other Area", (180, 180, 180)),
    ],
}



# ---------------------------------
# UI helper functions
# ---------------------------------

def create_mask_overlay(
    image_bgr,
    mask,
    overlay_color_bgr,
    alpha=0.55
):
    """
    Make the selected mask area stand out clearly:
    - masked region stays bright with colour tint
    - non-masked region becomes darker
    - contour boundary is drawn
    """

    if image_bgr is None:
        return None

    if mask is None:
        return image_bgr.copy()

    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8) * 255

    # Darken the whole image first
    darkened = (image_bgr * 0.25).astype(np.uint8)

    # Start from dark background
    output = darkened.copy()

    # Keep original region for selected area
    selected_region = image_bgr.copy()

    # Apply colour tint on selected area
    colour_layer = np.zeros_like(image_bgr)
    colour_layer[:] = overlay_color_bgr

    tinted = cv2.addWeighted(
        selected_region,
        1 - alpha,
        colour_layer,
        alpha,
        0
    )

    selected_pixels = binary_mask > 0
    output[selected_pixels] = tinted[selected_pixels]

    # Draw contour outline
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        output,
        contours,
        -1,
        (255, 255, 255),   # white border
        2
    )

    return output


def create_binary_mask_preview(mask):
    """
    Convert binary mask into a displayable 3-channel image.
    White = detected area
    Black = non-detected area
    """

    if mask is None:
        return None

    binary_mask = (mask > 0).astype(np.uint8) * 255

    return cv2.cvtColor(
        binary_mask,
        cv2.COLOR_GRAY2BGR
    )


def display_fruit_result(fruit):

    fruit_name = fruit["fruit_type"]

    st.subheader(
        f"Fruit {fruit['number']} - {fruit_name}"
    )

    # ========================================================
    # MAIN RIPENESS RESULT
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Ripeness Index",
            fruit["ripeness_index"]
        )

    with col2:
        st.metric(
            "Ripeness Stage",
            fruit["ripeness_label"]
        )

    with col3:
        confidence = fruit.get("confidence")

        if confidence is not None:
            st.metric(
                "Detection Confidence",
                f"{confidence * 100:.1f}%"
            )

    # ========================================================
    # DETECTED FRUIT CROP
    # ========================================================

    crop = fruit.get("crop")

    if crop is not None:
        st.image(
            crop,
            channels="BGR",
            caption=f"Detected {fruit_name}",
            use_container_width=True
        )

    # ========================================================
    # COLOUR PERCENTAGES
    # ========================================================

    details = fruit.get("details", {})

    if details:

        st.markdown("#### Colour Analysis")

        metric_columns = st.columns(
            min(len(details), 4)
        )

        for i, (name, value) in enumerate(details.items()):

            label = (
                name
                .replace("_percentage", "")
                .replace("_pct", "")
                .replace("_", " ")
                .title()
            )

            with metric_columns[
                i % len(metric_columns)
            ]:

                if isinstance(value, (int, float)):

                    st.metric(
                        label,
                        f"{value:.2f}%"
                    )

                else:

                    st.write(
                        f"**{label}:** {value}"
                    )

    # ========================================================
    # RIPENESS CLASSIFICATION RULE
    # ========================================================

    rule = fruit.get("rule_basis")

    if rule:

        with st.expander(
            "Classification Rule"
        ):
            st.write(rule)

    # ========================================================
    # DETECTED COLOUR AREAS
    # ========================================================

    diagnostics = fruit.get(
        "diagnostics",
        {}
    )

    if crop is not None and diagnostics:

        st.markdown(
            "#### Detected Colour Areas"
        )

        fruit_key = fruit_name.lower()

        configs = MASK_DISPLAY_CONFIG.get(
            fruit_key,
            []
        )

        available_masks = []

        for config in configs:

            mask_name = config[0]

            if mask_name in diagnostics:
                available_masks.append(config)

        if available_masks:

            # Display maximum 3 masks per row
            for start in range(
                0,
                len(available_masks),
                3
            ):

                current_masks = available_masks[
                    start:start + 3
                ]

                columns = st.columns(
                    len(current_masks)
                )

                for column, (
                    mask_name,
                    label,
                    display_colour
                ) in zip(
                    columns,
                    current_masks
                ):

                    mask = diagnostics[
                        mask_name
                    ]

                    overlay = create_mask_overlay(
                        crop,
                        mask,
                        display_colour
                    )

                    binary_preview = create_binary_mask_preview(
                        mask
                    )

                    isolated = create_isolated_region(
                        crop,
                        mask
                    )

                    with column:
                        st.markdown(f"**{label}**")

                        st.image(
                            overlay,
                            channels="BGR",
                            caption=f"{label} Overlay",
                            use_container_width=True
                        )

                        if binary_preview is not None:
                            st.image(
                                binary_preview,
                                channels="BGR",
                                caption=f"{label} Mask",
                                use_container_width=True
                            )

        else:

            st.info(
                "No colour masks available for this fruit."
            )

    st.divider()



def display_analysis_result(result):

    if result["count"] == 0:

        st.warning(
            "No fruit detected."
        )

        return

    st.success(
        f"{result['count']} fruit(s) detected."
    )

    # ========================================================
    # YOLO DETECTION RESULT
    # ========================================================

    st.markdown("### Fruit Detection")

    st.image(
        result["annotated_image"],
        channels="BGR",
        caption="Detected Fruits",
        use_container_width=True
    )

    # ========================================================
    # RIPENESS RESULTS
    # ========================================================

    st.markdown(
        "### Ripeness Assessment"
    )

    for fruit in result["fruits"]:
        display_fruit_result(fruit)



def create_isolated_region(image_bgr, mask):
    """
    Show only the detected area.
    Background becomes black.
    """

    if image_bgr is None or mask is None:
        return None

    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    binary_mask = (mask > 0).astype(np.uint8) * 255

    isolated = np.zeros_like(image_bgr)
    isolated[binary_mask > 0] = image_bgr[binary_mask > 0]

    return isolated


# ---------------------------------
# Parameter sidebar (modules 2.1 / 2.2 / 2.3)
# ---------------------------------

def render_param(param):
    key = param.key()
    if key not in st.session_state:
        st.session_state[key] = param.get()

    if param.kind == "bool":
        value = st.checkbox(param.label, key=key)
    elif param.kind == "text":
        value = st.text_input(param.label, key=key)
    elif param.kind == "int":
        value = st.slider(param.label, int(param.lo), int(param.hi), step=1, key=key)
    else:
        step = float(param.step) if param.step else 0.01
        value = st.slider(param.label, float(param.lo), float(param.hi),
                          step=step, key=key)
    param.set(value)


def render_sections(sections, prefix):
    for title, params in sections:
        with st.expander(title):
            for param in params:
                render_param(param)


def reset_sections(sections):
    for _, params in sections:
        for param in params:
            param.set(param.default)
            st.session_state.pop(param.key(), None)


def render_sidebar():
    st.sidebar.title("Parameters")

    tab1, tab2, tab3 = st.sidebar.tabs(["2.1 Pre", "2.2 Detect", "2.3 Ripeness"])

    with tab1:
        sections = preprocessing_sections()
        render_sections(sections, "p21")
        if st.button("Reset 2.1", key="reset21"):
            reset_sections(sections)
            st.rerun()

    with tab2:
        sections = detection_sections()
        render_sections(sections, "p22")
        resolved = detection.resolve_path(detection.PARAMS.model_path)
        if os.path.exists(resolved):
            st.caption(f"weights found: {detection.PARAMS.model_path}")
        else:
            st.error(f"weights NOT found: {detection.PARAMS.model_path}")
        if st.button("Reset 2.2", key="reset22"):
            reset_sections(sections)
            st.rerun()

    with tab3:
        fruit = st.selectbox("Fruit", list(ripeness.FRUIT_MODULES), key="ripe_fruit")
        sections = ripeness_sections(fruit)
        render_sections(sections, f"p23_{fruit}")
        if st.button(f"Reset {fruit}", key=f"reset23_{fruit}"):
            reset_sections(sections)
            st.rerun()


# ---------------------------------
# Pipeline (modules 2.1 -> 2.2)
# ---------------------------------

def build_crop_grid(crops, panel=380):
    if not crops:
        return None
    cols = 1 if len(crops) == 1 else 2
    rows = 1 if len(crops) <= 2 else 2
    tw, th = panel // cols, panel // rows
    grid = np.full((th * rows, tw * cols, 3), 238, dtype=np.uint8)
    for i, crop in enumerate(crops[:cols * rows]):
        h, w = crop.shape[:2]
        s = min(tw / w, th / h)
        rw, rh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(crop, (rw, rh))
        r, col = divmod(i, cols)
        y0 = r * th + (th - rh) // 2
        x0 = col * tw + (tw - rw) // 2
        grid[y0:y0 + rh, x0:x0 + rw] = small
    return grid


def run_pipeline(image_bgr):
    prep = preprocessing.preprocess(image_bgr)

    if not prep.crops:
        crops = [prep.bgr]
        crop_rects = [(0, 0, prep.bgr.shape[1], prep.bgr.shape[0])]
    else:
        crops = prep.crops
        crop_rects = prep.crop_rects

    model = detection.load_model()
    detections = segment(model, crops, crop_rects, image_bgr.shape)
    count = contour_count(detections)

    prepared = []
    for i, d in enumerate(detections, start=1):
        crop, mask, masked = detection.crop_detection_from_original(image_bgr, d)
        prepared.append({
            "number": i,
            "yolo_fruit_type": detection.CLASS_NAMES[int(d["cls"])],
            "confidence": float(d["conf"]),
            "crop": crop,
            "fruit_mask": mask,
            "masked_crop": masked,
        })

    return {
        "image": image_bgr,
        "crops": crops,
        "detections": detections,
        "prepared": prepared,
        "count": count,
    }


def display_stages(result, show_mask):
    image = result["image"]
    detections = result["detections"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Original**")
        st.image(image, channels="BGR", use_container_width=True)
    with col2:
        st.markdown(f"**2. Preprocessed** ({len(result['crops'])} object crops)")
        grid = build_crop_grid(result["crops"])
        if grid is not None:
            st.image(grid, channels="BGR", use_container_width=True)
    with col3:
        st.markdown("**3. Segmented + Contours**")
        st.image(annotate(image, detections, show_mask=show_mask),
                 channels="BGR", use_container_width=True)

    st.info(f"Contour count (fruits detected): {result['count']}")

    if detections:
        summary = {}
        for d in detections:
            name = detection.CLASS_NAMES[d["cls"]]
            summary[name] = summary.get(name, 0) + 1
        st.write("**Fruit summary:** " +
                 ", ".join(f"{k}: {v}" for k, v in sorted(summary.items())))
        rows = [{"Fruit": i + 1,
                 "Type": detection.CLASS_NAMES[d["cls"]],
                 "Confidence": round(d["conf"], 3),
                 "Contour area (px)": round(d.get("area", 0), 1)}
                for i, d in enumerate(detections)]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def per_instance_view(result, show_mask):
    detections = result["detections"]
    if not detections:
        return
    labels = ["All"] + [f"{i + 1}: {detection.CLASS_NAMES[d['cls']]}"
                        for i, d in enumerate(detections)]
    choice = st.selectbox("Inspect instance", labels, key="seg_instance")
    if choice == "All":
        return
    index = labels.index(choice) - 1
    d = detections[index]
    annotated = annotate(result["image"], [d], show_mask=show_mask)
    x, y, w, h = cv2.boundingRect(d["mask"])
    if w and h:
        pad = int(max(w, h) * 0.15)
        H, W = annotated.shape[:2]
        annotated = annotated[max(0, y - pad):min(H, y + h + pad),
                              max(0, x - pad):min(W, x + w + pad)]
    st.image(annotated, channels="BGR", use_container_width=False)


def export_section(result):
    if not result["detections"]:
        return
    if not st.button("Export masked crops for ripeness", key="export"):
        return
    session, manifest = export_for_ripeness(result["image"], result["detections"])
    st.success(f"Exported {len(manifest)} masked crop(s) to {session}")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in os.listdir(session):
            archive.write(os.path.join(session, name), name)
    st.download_button("Download export as .zip", buffer.getvalue(),
                       file_name=f"{os.path.basename(session)}.zip",
                       mime="application/zip")


# ---------------------------------
# Training
# ---------------------------------

def training_section():
    st.markdown("### Model training")
    include = st.checkbox("Include false images", value=training.PARAMS.include_negatives,
                          key="train_negatives")
    training.PARAMS.include_negatives = include

    col1, col2, col3 = st.columns(3)
    with col1:
        training.PARAMS.epochs = st.number_input("Epochs", 1, 1000,
                                                 training.PARAMS.epochs, key="train_epochs")
    with col2:
        training.PARAMS.batch = st.number_input("Batch", 1, 64,
                                                training.PARAMS.batch, key="train_batch")
    with col3:
        training.PARAMS.workers = st.number_input("Workers", 0, 16,
                                                  training.PARAMS.workers, key="train_workers")

    run_name = training.run_name_for()
    st.caption(f"Dataset: {training.PARAMS.dataset_dir} -> "
               f"run: {training.PARAMS.runs_dir}/{run_name}")
    st.warning("This rebuilds the dataset folder and overwrites that run folder, "
               "including its current best.pt.")

    if not st.button("Build dataset and train", key="train_go", type="primary"):
        return

    weights = f"{training.PARAMS.runs_dir}/{run_name}/weights/best.pt"
    command = [
        sys.executable, "-u", "training.py",
        "--negatives", "1" if include else "0",
        "--epochs", str(training.PARAMS.epochs),
        "--batch", str(training.PARAMS.batch),
        "--workers", str(training.PARAMS.workers),
        "--imgsz", str(training.PARAMS.imgsz),
        "--device", str(training.PARAMS.device),
    ]

    log_box = st.empty()
    lines = []
    with st.spinner("Training..."):
        process = subprocess.Popen(
            command, cwd=str(training.PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            lines.append(line)
            log_box.code("\n".join(lines[-25:]))
        code = process.wait()

    if code == 0:
        st.success(f"Training complete: {weights}")
        if os.path.exists(detection.resolve_path(weights)):
            detection.PARAMS.model_path = weights
            st.session_state.pop(
                f"{detection.PARAMS.__class__.__name__}.model_path.()", None)
            st.info("Detection now uses the newly trained weights.")
    else:
        st.error(f"training.py exited with code {code}")


# ---------------------------------
# Main
# ---------------------------------

render_sidebar()

st.title("Fruit Ripeness Assessment")

tab_analyse, tab_train = st.tabs(["Analyse", "Train"])

with tab_train:
    training_section()

with tab_analyse:
    source = st.radio("Input", ["Upload image", "Camera"], horizontal=True, key="source")

    image_bgr = None
    source_name = None

    if source == "Upload image":
        uploaded_file = st.file_uploader("Choose a fruit image",
                                         type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file is not None:
            data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
            image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            source_name = uploaded_file.name
    else:
        shot = st.camera_input("Take a picture")
        if shot is not None:
            data = np.frombuffer(shot.getvalue(), dtype=np.uint8)
            image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            source_name = "camera"

    if image_bgr is None:
        st.info("Upload an image or take a picture to begin.")
    else:
        if st.session_state.get("last_source") != source_name:
            st.session_state["last_source"] = source_name
            st.session_state.pop("pipeline_result", None)
            st.session_state.pop("ripeness_result", None)

        show_mask = st.checkbox("Show mask overlay", value=True, key="show_mask")

        if st.button("Run pipeline", type="primary", key="run"):
            try:
                with st.spinner("Running modules 2.1 and 2.2..."):
                    st.session_state["pipeline_result"] = run_pipeline(image_bgr)
                st.session_state.pop("ripeness_result", None)
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")

        result = st.session_state.get("pipeline_result")
        if result is not None:
            display_stages(result, show_mask)
            per_instance_view(result, show_mask)
            export_section(result)

            if result["count"] == 0:
                st.warning("No fruit detected.")
            else:
                st.markdown("### Confirm fruit type")
                labels = [f"{d['number']}: {d['yolo_fruit_type']} "
                          f"({d['confidence'] * 100:.1f}%)" for d in result["prepared"]]
                picked = st.selectbox("Detected fruit", labels, key="pick_fruit")
                detected = result["prepared"][labels.index(picked)]

                options = [n.title() for n in detection.CLASS_NAMES]
                predicted = detected["yolo_fruit_type"].title()
                selected = st.selectbox(
                    "If YOLO is wrong, select the correct fruit:",
                    options,
                    index=options.index(predicted) if predicted in options else 0,
                    key="override_fruit")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Continue with {predicted}", type="primary",
                                 use_container_width=True, key="go_yolo"):
                        try:
                            with st.spinner(f"Analyzing {predicted} ripeness..."):
                                st.session_state["ripeness_result"] = \
                                    assess_detected_fruit(detected)
                        except Exception as exc:
                            st.error(f"Ripeness assessment failed: {exc}")
                with col2:
                    if st.button(f"Analyze as {selected}",
                                 disabled=selected == predicted,
                                 use_container_width=True, key="go_override"):
                        try:
                            with st.spinner(f"Analyzing {selected} ripeness..."):
                                st.session_state["ripeness_result"] = \
                                    assess_detected_fruit(detected,
                                                          selected_fruit_type=selected)
                        except Exception as exc:
                            st.error(f"Ripeness assessment failed: {exc}")

                if st.button("Assess ALL detected fruits", key="go_all"):
                    results = []
                    for d in result["prepared"]:
                        try:
                            results.append(assess_detected_fruit(d))
                        except Exception as exc:
                            st.error(f"Fruit {d['number']} failed: {exc}")
                    st.session_state["ripeness_all"] = results
                    st.session_state.pop("ripeness_result", None)

        if "ripeness_result" in st.session_state:
            st.markdown("## Ripeness Assessment")
            display_fruit_result(st.session_state["ripeness_result"])

        if "ripeness_all" in st.session_state:
            st.markdown("## Ripeness Assessment - all fruits")
            for item in st.session_state["ripeness_all"]:
                display_fruit_result(item)
