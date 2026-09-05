"""Module 2.2 parameter panel - Fruit Detection and Instance Segmentation."""

import os
from pathlib import Path
from tkinter import filedialog, ttk

import detection
from gui.param_widgets import ParamEditor, ScrollableFrame
from params_spec import detection_sections


def build(parent, on_change, on_model_change=None):
    editor = ParamEditor(on_change)

    area = ScrollableFrame(parent)
    area.pack(fill="both", expand=True)
    body = area.body
    body.columnconfigure(0, weight=1)

    model_param = None
    row = 0
    for title, params in detection_sections():
        frame = editor.section(body, title, params, row)
        if title == "Model":
            model_param = params[0]

            def browse():
                path = filedialog.askopenfilename(
                    title="Select YOLO weights",
                    initialdir=os.path.dirname(
                        detection.resolve_path(detection.PARAMS.model_path)) or ".",
                    filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")])
                if not path:
                    return
                try:
                    path = Path(path).resolve().relative_to(
                        detection.PROJECT_ROOT).as_posix()
                except ValueError:
                    pass
                model_param.set(path)
                editor.refresh()
                if on_model_change is not None:
                    on_model_change()
                on_change()

            ttk.Button(frame, text="Browse...", command=browse).grid(
                row=len(params), column=1, columnspan=2, sticky="e", pady=(4, 0))
        row += 1

    ttk.Button(body, text="Reset 2.2 defaults", command=editor.reset).grid(
        row=row, column=0, sticky="w", padx=4, pady=6)

    return editor
