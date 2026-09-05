import streamlit as st
import time

import cv2
import numpy as np

from Backend.camera import Camera
from Backend.ripeness_service import (
    detect_image,
    assess_detected_fruit
)


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
# existing Streamlit UI
# ---------------------------------

st.title("Camera Feed")

run = st.checkbox("Start Camera")

frame_placeholder = st.empty()

camera = Camera(0)

while run:
    frame = camera.get_frame()

    if frame is None:
        st.error("Failed to access camera.")
        break

    frame_placeholder.image(frame, channels="RGB")

    time.sleep(0.03)

camera.release()



# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("## Fruit Ripeness Assessment")

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    current_file = uploaded_file.name

    # Clear old result when another file is selected
    if (
        "last_uploaded_file" not in st.session_state
        or
        st.session_state["last_uploaded_file"] != current_file
    ):

        st.session_state[
            "last_uploaded_file"
        ] = current_file

        st.session_state.pop(
            "detection_result",
            None
        )

        st.session_state.pop(
            "ripeness_result",
            None
        )

    # --------------------------------------------------------
    # Convert uploaded image to OpenCV BGR
    # --------------------------------------------------------

    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8
    )

    image_bgr = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:

        st.error(
            "Unable to read selected image."
        )

    else:

        st.markdown("### Selected Image")

        st.image(
            image_bgr,
            channels="BGR",
            caption=uploaded_file.name,
            use_container_width=True
        )

        # ====================================================
        # STEP 1 - YOLO DETECTION
        # ====================================================

        if st.button(
            "Detect Fruit",
            type="primary"
        ):

            with st.spinner(
                "Detecting fruit..."
            ):

                try:

                    detection_result = detect_image(
                        image_bgr
                    )

                    st.session_state[
                        "detection_result"
                    ] = detection_result

                    st.session_state.pop(
                        "ripeness_result",
                        None
                    )

                except Exception as e:

                    st.error(
                        f"Fruit detection failed: {e}"
                    )

# ============================================================
# STEP 2 - CONFIRM YOLO FRUIT TYPE
# ============================================================

if "detection_result" in st.session_state:

    detection_result = st.session_state[
        "detection_result"
    ]

    if detection_result["count"] == 0:

        st.warning(
            "No fruit detected."
        )

    else:

        st.markdown("## Fruit Detection")

        st.image(
            detection_result[
                "annotated_image"
            ],
            channels="BGR",
            caption="YOLO Fruit Detection",
            use_container_width=True
        )

        detected_fruit = (
            detection_result[
                "detections"
            ][0]
        )

        predicted_fruit = (
            detected_fruit[
                "yolo_fruit_type"
            ].title()
        )

        confidence = detected_fruit[
            "confidence"
        ]

        st.info(
            f"YOLO predicts: "
            f"**{predicted_fruit}** "
            f"({confidence * 100:.1f}% confidence)"
        )

        st.markdown(
            "### Is this fruit correct?"
        )

        fruit_options = [
            "Apple",
            "Banana",
            "Mango",
            "Orange",
            "Strawberry",
            "Tomato",
        ]

        default_index = (
            fruit_options.index(
                predicted_fruit
            )
            if predicted_fruit
            in fruit_options
            else 0
        )

        selected_fruit = st.selectbox(
            "If YOLO is wrong, select the correct fruit:",
            fruit_options,
            index=default_index,
        )

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # Continue with YOLO result
        # ----------------------------------------------------

        with col1:

            if st.button(
                f"Continue with {predicted_fruit}",
                type="primary",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        f"Analyzing {predicted_fruit} ripeness..."
                    ):

                        fruit_result = (
                            assess_detected_fruit(
                                detected_fruit
                            )
                        )

                    st.session_state[
                        "ripeness_result"
                    ] = fruit_result

                except Exception as e:

                    st.error(
                        f"Ripeness assessment failed: {e}"
                    )

        # ----------------------------------------------------
        # Override YOLO result
        # ----------------------------------------------------

        override_disabled = (
            selected_fruit == predicted_fruit
        )

        with col2:

            if st.button(
                f"Analyze as {selected_fruit}",
                disabled=override_disabled,
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        f"Analyzing {selected_fruit} ripeness..."
                    ):

                        fruit_result = (
                            assess_detected_fruit(
                                detected_fruit,
                                selected_fruit_type=selected_fruit
                            )
                        )

                    st.session_state[
                        "ripeness_result"
                    ] = fruit_result

                except Exception as e:

                    st.error(
                        f"Ripeness assessment failed: {e}"
                    )


# ============================================================
# FINAL RIPENESS RESULT
# ============================================================

if "ripeness_result" in st.session_state:

    st.markdown(
        "## Ripeness Assessment"
    )

    fruit_result = st.session_state[
        "ripeness_result"
    ]

    display_fruit_result(
        fruit_result
    )