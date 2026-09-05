"""Module 2.2 parameter panel - Fruit Detection and Instance Segmentation."""

import os
from tkinter import filedialog, ttk

import detection
from gui.param_widgets import Param, ParamEditor, ScrollableFrame


def build(parent, on_change, on_model_change=None):
    p = detection.PARAMS
    editor = ParamEditor(on_change)

    area = ScrollableFrame(parent)
    area.pack(fill="both", expand=True)
    body = area.body
    body.columnconfigure(0, weight=1)

    model_frame = ttk.LabelFrame(body, text="Model", padding=6)
    model_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
    model_frame.columnconfigure(1, weight=1)

    model_param = Param("Weights", p, "model_path", kind="text")
    editor.add(model_frame, model_param, 0)

    def browse():
        path = filedialog.askopenfilename(
            title="Select YOLO weights",
            initialdir=os.path.dirname(p.model_path) or ".",
            filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        model_param.set(path)
        editor.refresh()
        if on_model_change is not None:
            on_model_change()
        on_change()

    ttk.Button(model_frame, text="Browse...", command=browse).grid(
        row=1, column=1, columnspan=2, sticky="e", pady=(4, 0))

    editor.section(body, "Segmentation", [
        Param("Confidence", p, "conf_threshold", kind="float", lo=0.01, hi=1.0),
        Param("Mask binary thr.", p, "mask_binary_threshold", kind="float", lo=0.01, hi=0.99),
        Param("Dedupe overlap", p, "dedupe_overlap_threshold", kind="float", lo=0.0, hi=1.0),
        Param("Retina masks", p, "retina_masks", kind="bool"),
    ], 1)

    editor.section(body, "Annotation", [
        Param("Overlay alpha", p, "overlay_alpha", kind="float", lo=0.0, hi=1.0),
        Param("Contour thickness", p, "contour_thickness", kind="int", lo=1, hi=10),
        Param("Label font scale", p, "label_font_scale", kind="float", lo=0.2, hi=3.0),
    ], 2)

    editor.section(body, "Export", [
        Param("Output folder", p, "output_dir", kind="text"),
    ], 3)

    ttk.Button(body, text="Reset 2.2 defaults", command=editor.reset).grid(
        row=4, column=0, sticky="w", padx=4, pady=6)

    return editor
