import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

MASK_DISPLAY_CONFIG = {
    "apple": [
        ("red_mask", "Red Area", (0, 0, 255), "red_percentage"),
        ("yellow_mask", "Yellow Area", (0, 255, 255), "yellow_percentage"),
        ("other_mask", "Other Area", (180, 180, 180), "other_percentage"),
    ],
    "banana": [
        ("yellow_mask", "Yellow Area", (0, 255, 255), "yellow_percentage"),
        ("brown_mask", "Brown Area", (42, 42, 165), "brown_percentage"),
        (
            "developed_peel_mask",
            "Developed Peel Area",
            (0, 165, 255),
            "developed_peel_percentage",
        ),
        ("other_mask", "Other Area", (180, 180, 180), "other_percentage"),
    ],
    "mango": [
        ("yellow_mask", "Yellow Area", (0, 255, 255), "yellow_percentage"),
        ("other_mask", "Other Area", (180, 180, 180), "other_percentage"),
    ],
    "orange": [
        (
            "yellow_orange_mask",
            "Yellow / Orange Area",
            (0, 165, 255),
            "yellow_orange_percentage",
        ),
        (
            "non_yellow_orange_mask",
            "Non Yellow / Orange Area",
            (180, 180, 180),
            "non_yellow_orange_percentage",
        ),
    ],
    "strawberry": [
        ("red_mask", "Red Area", (0, 0, 255), "red_percentage"),
        ("dark_red_mask", "Dark Red Area", (0, 0, 120), "dark_red_percentage"),
    ],
    "tomato": [
        ("green_mask", "Green Area", (0, 255, 0), "green_percentage"),
        ("turning_mask", "Turning Area", (0, 255, 255), "turning_percentage"),
        ("pink_mask", "Pink Area", (180, 105, 255), "pink_percentage"),
        ("mature_red_mask", "Mature Red Area", (0, 0, 255), "mature_red_percentage"),
        (
            "colour_break_mask",
            "Colour Break Area",
            (0, 200, 255),
            "colour_break_percentage",
        ),
        (
            "red_orange_mask",
            "Red / Orange Area",
            (0, 100, 255),
            "red_orange_percentage",
        ),
        ("other_mask", "Other Area", (180, 180, 180), "other_percentage"),
    ],
}


def create_mask_overlay(image_bgr, mask, overlay_colour_bgr, alpha=0.55):
    if image_bgr is None or mask is None:
        return None

    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    binary_mask = (mask > 0).astype(np.uint8) * 255

    output = (image_bgr * 0.25).astype(np.uint8)

    colour_layer = np.zeros_like(image_bgr)
    colour_layer[:] = overlay_colour_bgr

    tinted = cv2.addWeighted(image_bgr, 1 - alpha, colour_layer, alpha, 0)

    selected_pixels = binary_mask > 0
    output[selected_pixels] = tinted[selected_pixels]

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(output, contours, -1, (255, 255, 255), 2)

    return output


def create_isolated_region(image_bgr, mask):
    if image_bgr is None or mask is None:
        return None

    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    binary_mask = (mask > 0).astype(np.uint8) * 255

    isolated = np.zeros_like(image_bgr)
    isolated[binary_mask > 0] = image_bgr[binary_mask > 0]

    return isolated


def create_binary_mask_preview(mask):
    if mask is None:
        return None

    binary_mask = (mask > 0).astype(np.uint8) * 255

    return cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)


def format_detail(name, value):
    label = (
        name.replace("_percentage", "").replace("_pct", "").replace("_", " ").title()
    )

    if not isinstance(value, (int, float)):
        return label, str(value)

    if "percentage" in name or name.endswith("_pct"):
        return label, f"{value:.2f}%"

    return label, f"{value:.2f}"


def cv_to_tk_image(image_bgr, max_width=260, max_height=220):
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    h, w = image_rgb.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)

    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return ImageTk.PhotoImage(Image.fromarray(image_rgb))


def show_ripeness_result(parent, result):
    popup = tk.Toplevel(parent)
    popup.title("Ripeness Assessment Result")
    popup.geometry("1100x800")
    popup.transient(parent)

    popup.summary_image_refs = []
    popup.colour_image_refs = []

    ttk.Label(
        popup,
        text=f"{result['fruit_type']} Ripeness Assessment",
        font=("Arial", 18, "bold"),
    ).pack(pady=(15, 8))

    yolo_type = result.get("yolo_fruit_type")
    selected_type = result.get("selected_fruit_type")

    if yolo_type and selected_type and yolo_type.lower() != selected_type.lower():
        ttk.Label(
            popup,
            text=f"YOLO predicted {yolo_type.title()} — manually analysed as {selected_type.title()}",
            foreground="dark orange",
        ).pack(pady=(0, 5))

    notebook = ttk.Notebook(popup)
    notebook.pack(fill="both", expand=True, padx=15, pady=10)

    summary_tab = ttk.Frame(notebook)
    colour_tab = ttk.Frame(notebook)

    notebook.add(summary_tab, text="Summary")
    notebook.add(colour_tab, text="Detected Colour Areas")

    result_frame = ttk.Frame(summary_tab)
    result_frame.pack(fill="x", padx=20, pady=15)

    result_frame.columnconfigure(0, weight=1)
    result_frame.columnconfigure(1, weight=1)
    result_frame.columnconfigure(2, weight=1)

    ttk.Label(
        result_frame,
        text=f"Ripeness Index\n{result['ripeness_index']}",
        font=("Arial", 13, "bold"),
        justify="center",
    ).grid(row=0, column=0, padx=25, pady=5)
    ttk.Label(
        result_frame,
        text=f"Ripeness Stage\n{result['ripeness_label']}",
        font=("Arial", 13, "bold"),
        justify="center",
    ).grid(row=0, column=1, padx=25, pady=5)

    confidence = result.get("confidence")

    if confidence is not None:
        ttk.Label(
            result_frame,
            text=f"YOLO Confidence\n{confidence * 100:.1f}%",
            font=("Arial", 13),
            justify="center",
        ).grid(row=0, column=2, padx=25, pady=5)

    crop = result.get("crop")

    if crop is not None:
        ttk.Label(summary_tab, text="Detected Fruit", font=("Arial", 13, "bold")).pack(
            pady=(8, 5)
        )

        crop_tk = cv_to_tk_image(crop, max_width=380, max_height=260)

        popup.summary_image_refs.append(crop_tk)

        ttk.Label(summary_tab, image=crop_tk).pack(pady=5)

    ttk.Label(summary_tab, text="Colour Analysis", font=("Arial", 14, "bold")).pack(
        pady=(15, 10)
    )

    details_frame = ttk.Frame(summary_tab)
    details_frame.pack(pady=5)

    details = result.get("details", {})

    for column, (name, value) in enumerate(details.items()):
        label, formatted_value = format_detail(name, value)

        ttk.Label(
            details_frame,
            text=f"{label}\n{formatted_value}",
            font=("Arial", 12),
            justify="center",
        ).grid(row=0, column=column, padx=22, pady=5)

    rule = result.get("rule_basis")

    if rule:
        ttk.Label(
            summary_tab, text="Classification Rule", font=("Arial", 13, "bold")
        ).pack(pady=(20, 5))
        ttk.Label(summary_tab, text=str(rule), wraplength=850, justify="center").pack(
            padx=20, pady=5
        )

    diagnostics = result.get("diagnostics", {})
    fruit_key = result["fruit_type"].lower()

    configs = MASK_DISPLAY_CONFIG.get(fruit_key, [])

    available_masks = [config for config in configs if config[0] in diagnostics]

    if not available_masks:
        ttk.Label(
            colour_tab, text="No colour masks are available for this fruit."
        ).pack(pady=30)

    else:
        mode_frame = ttk.Frame(colour_tab)
        mode_frame.pack(pady=(12, 5))

        ttk.Label(mode_frame, text="Display Mode:", font=("Arial", 11, "bold")).pack(
            side="left", padx=(0, 10)
        )

        display_container = ttk.Frame(colour_tab)
        display_container.pack(fill="both", expand=True, padx=10, pady=5)

        canvas = tk.Canvas(display_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            display_container, orient="vertical", command=canvas.yview
        )

        display_frame = ttk.Frame(canvas)

        canvas_window = canvas.create_window((0, 0), window=display_frame, anchor="nw")

        display_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(canvas_window, width=event.width),
        )

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def refresh_colour_view(mode):
            for widget in display_frame.winfo_children():
                widget.destroy()

            popup.colour_image_refs.clear()

            mode_titles = {
                "highlight": "Highlighted Area",
                "isolated": "Isolated Area",
                "binary": "Binary Mask",
            }

            mode_title = mode_titles.get(mode, "Highlighted Area")

            for index, (mask_name, label, display_colour, percentage_key) in enumerate(
                available_masks
            ):
                mask = diagnostics[mask_name]

                if mode == "highlight":
                    display_image = create_mask_overlay(crop, mask, display_colour)

                elif mode == "isolated":
                    display_image = create_isolated_region(crop, mask)

                elif mode == "binary":
                    display_image = create_binary_mask_preview(mask)

                else:
                    continue

                percentage = details.get(percentage_key)

                if isinstance(percentage, (int, float)):
                    frame_title = f"{label} — {percentage:.2f}%"
                else:
                    frame_title = label

                colour_frame = ttk.LabelFrame(display_frame, text=frame_title)
                colour_frame.grid(
                    row=index // 3, column=index % 3, padx=10, pady=10, sticky="n"
                )

                display_frame.columnconfigure(index % 3, weight=1)

                ttk.Label(
                    colour_frame, text=mode_title, font=("Arial", 10, "bold")
                ).pack(pady=(8, 5))

                if display_image is not None:
                    image_tk = cv_to_tk_image(
                        display_image, max_width=280, max_height=260
                    )

                    popup.colour_image_refs.append(image_tk)

                    ttk.Label(colour_frame, image=image_tk).pack(padx=8, pady=(5, 10))

        ttk.Button(
            mode_frame,
            text="Highlighted Area",
            command=lambda: refresh_colour_view("highlight"),
        ).pack(side="left", padx=5)
        ttk.Button(
            mode_frame,
            text="Isolated Area",
            command=lambda: refresh_colour_view("isolated"),
        ).pack(side="left", padx=5)
        ttk.Button(
            mode_frame,
            text="Binary Mask",
            command=lambda: refresh_colour_view("binary"),
        ).pack(side="left", padx=5)

        refresh_colour_view("highlight")

    ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=(0, 15))
