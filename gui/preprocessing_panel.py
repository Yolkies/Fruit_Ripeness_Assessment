"""Module 2.1 parameter panel - Image Pre-processing and Classical Segmentation."""

from tkinter import ttk

import preprocessing
from gui.param_widgets import Param, ParamEditor, ScrollableFrame


def build(parent, on_change):
    p = preprocessing.PARAMS
    editor = ParamEditor(on_change)

    area = ScrollableFrame(parent)
    area.pack(fill="both", expand=True)
    body = area.body
    body.columnconfigure(0, weight=1)

    editor.section(body, "1.1 Resize / normalise", [
        Param("Enabled", p, "resize_enabled", kind="bool"),
        Param("Target long side", p, "target_long_side", kind="int", lo=128, hi=2048),
    ], 0)

    editor.section(body, "1.2 White balance", [
        Param("Enabled", p, "white_balance_enabled", kind="bool"),
        Param("Saturation thr.", p, "wb_saturation_threshold", kind="float", lo=0.0, hi=1.0),
        Param("Strength", p, "wb_strength", kind="float", lo=0.0, hi=1.0),
        Param("Max gain", p, "wb_max_gain", kind="float", lo=1.0, hi=4.0),
    ], 1)

    editor.section(body, "1.3 Denoise", [
        Param("Enabled", p, "denoise_enabled", kind="bool"),
        Param("Median ksize (odd)", p, "median_ksize", kind="int", lo=1, hi=15),
        Param("Bilateral d", p, "bilateral_d", kind="int", lo=1, hi=15),
        Param("Sigma colour", p, "bilateral_sigma_colour", kind="int", lo=1, hi=200),
        Param("Sigma space", p, "bilateral_sigma_space", kind="int", lo=1, hi=200),
    ], 2)

    editor.section(body, "1.4 CLAHE", [
        Param("Enabled", p, "clahe_enabled", kind="bool"),
        Param("Clip limit", p, "clahe_clip_limit", kind="float", lo=0.1, hi=10.0),
        Param("Tile grid", p, "clahe_tile_grid", kind="int", lo=1, hi=32),
        Param("Contrast gate", p, "clahe_contrast_gate", kind="float", lo=0.0, hi=128.0),
    ], 3)

    editor.section(body, "1.5 Glare removal", [
        Param("Enabled", p, "glare_enabled", kind="bool"),
        Param("V min", p, "glare_v_min", kind="int", lo=0, hi=255),
        Param("S max", p, "glare_s_max", kind="int", lo=0, hi=255),
        Param("Inpaint radius", p, "glare_inpaint_radius", kind="int", lo=1, hi=25),
    ], 4)

    editor.section(body, "1.6 Otsu cropping", [
        Param("Min area (px)", p, "otsu_min_area", kind="int", lo=0, hi=50000),
        Param("Pad percent", p, "otsu_pad_percent", kind="float", lo=0.0, hi=1.0),
        Param("Blur ksize (odd)", p, "otsu_blur_ksize", kind="int", lo=1, hi=51),
        Param("Morph ksize", p, "otsu_morph_ksize", kind="int", lo=1, hi=51),
        Param("Dilate ksize", p, "otsu_dilate_ksize", kind="int", lo=1, hi=51),
    ], 5)

    ttk.Button(body, text="Reset 2.1 defaults", command=editor.reset).grid(
        row=6, column=0, sticky="w", padx=4, pady=6)

    return editor
