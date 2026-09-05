"""Module 2.3 parameter panel - Colour-Based Fruit Ripeness Assessment."""

from tkinter import ttk

import ripeness
from gui.param_widgets import (
    Param,
    ParamEditor,
    ScrollableFrame,
    hsv_range_pair,
    hsv_triplet,
)


def _hue_range(label, module, name, path=()):
    return [
        Param(f"{label} hue lo", module, name, path + (0,), "int", 0, 179, 1),
        Param(f"{label} hue hi", module, name, path + (1,), "int", 0, 179, 1),
    ]


def _pct(label, module, name):
    return Param(label, module, name, kind="float", lo=0.0, hi=100.0, step=0.5)


def _sections(fruit, module):
    """(title, [Param, ...]) groups for one fruit's tunable constants."""
    if fruit == "apple":
        return [
            ("Colour thresholds",
             _hue_range("Red band 1", module, "RED_HUE_RANGES", (0,))
             + _hue_range("Red band 2", module, "RED_HUE_RANGES", (1,))
             + _hue_range("Yellow", module, "YELLOW_HUE_RANGE")),
            ("Maturity boundaries (% red)", [
                _pct("Index 2 min", module, "INDEX2_MIN"),
                _pct("Index 3 min", module, "INDEX3_MIN"),
                _pct("Index 4 min", module, "INDEX4_MIN"),
                _pct("Index 5 min", module, "INDEX5_MIN"),
            ]),
        ]

    if fruit == "banana":
        return [
            ("Yellow mask",
             hsv_triplet("Lower", module, "YELLOW_LOWER")
             + hsv_triplet("Upper", module, "YELLOW_UPPER")),
            ("Brown mask",
             hsv_triplet("Lower", module, "BROWN_LOWER")
             + hsv_triplet("Upper", module, "BROWN_UPPER")),
            ("Ripeness boundaries (%)", [
                _pct("Index 1 yellow noise max", module, "INDEX1_YELLOW_NOISE_MAX"),
                _pct("Complete yellow min", module, "COMPLETE_YELLOW_MIN"),
                _pct("Brown spot min", module, "BROWN_SPOT_MIN"),
                _pct("Heavy brown overripe min", module, "HEAVY_BROWN_OVERRIPE_MIN"),
                _pct("Developed peel min", module, "DEVELOPED_PEEL_MIN"),
            ]),
        ]

    if fruit == "mango":
        return [
            ("Yellow mask",
             hsv_triplet("Lower", module, "YELLOW_LOWER")
             + hsv_triplet("Upper", module, "YELLOW_UPPER")),
            ("Stage boundaries (% yellow)", [
                _pct("Stage 1 max", module, "STAGE1_MAX"),
                _pct("Stage 2 max", module, "STAGE2_MAX"),
                _pct("Stage 3 max", module, "STAGE3_MAX"),
            ]),
        ]

    if fruit == "orange":
        return [
            ("Maturity boundaries (% yellow/orange)", [
                _pct("Index 2 min", module, "INDEX2_MIN"),
                _pct("Index 3 min", module, "INDEX3_MIN"),
                _pct("Index 4 min", module, "INDEX4_MIN"),
                _pct("Index 5 min", module, "INDEX5_MIN"),
            ]),
        ]

    if fruit == "strawberry":
        return [
            ("Red mask",
             hsv_range_pair("Band 1", module, "RED_RANGES", (0,))
             + hsv_range_pair("Band 2", module, "RED_RANGES", (1,))),
            ("Dark red mask",
             hsv_range_pair("Band 1", module, "DARK_RED_RANGES", (0,))
             + hsv_range_pair("Band 2", module, "DARK_RED_RANGES", (1,))),
            ("Maturity boundaries (%)", [
                _pct("FAMA red boundary", module, "FAMA_RED_BOUNDARY"),
                _pct("Dark red overripe", module, "FUKUDA_DARK_RED_OVERRIPE"),
                _pct("Supplementary unripe red max", module,
                     "SUPPLEMENTARY_UNRIPE_RED_MAX"),
            ]),
            ("Rules", [
                Param("Supplementary Index-1 rule", ripeness.PARAMS,
                      "strawberry_supplementary_index1_rule", kind="bool"),
            ]),
        ]

    if fruit == "tomato":
        return [
            ("Green mask", hsv_range_pair("Green", module, "GREEN_RANGE")),
            ("Turning mask", hsv_range_pair("Turning", module, "TURNING_RANGE")),
            ("Pink mask", hsv_range_pair("Pink", module, "PINK_RANGE")),
            ("Red mask",
             hsv_range_pair("Band 1", module, "RED_RANGES", (0,))
             + hsv_range_pair("Band 2", module, "RED_RANGES", (1,))),
            ("Maturity boundaries (%)", [
                _pct("Early green colour-break max", module,
                     "EARLY_GREEN_COLOUR_BREAK_MAX"),
                _pct("Index 3 boundary", module, "INDEX3_BOUNDARY"),
                _pct("Index 4 boundary", module, "INDEX4_BOUNDARY"),
                _pct("Full red boundary", module, "FULL_RED_BOUNDARY"),
            ]),
            ("L* thresholds", [
                Param("Dark L* threshold", module, "DARK_LSTAR_THRESHOLD",
                      kind="float", lo=0.0, hi=100.0),
            ]),
        ]

    return []


class _AllFruits:
    """Aggregates the per-fruit editors so the caller can treat them as one."""

    def __init__(self, editors):
        self.editors = editors

    def refresh(self):
        for editor in self.editors:
            editor.refresh()

    def reset(self):
        for editor in self.editors:
            editor.reset()


def build(parent, on_change):
    editors = []

    book = ttk.Notebook(parent)
    book.pack(fill="both", expand=True)

    for fruit, module in ripeness.FRUIT_MODULES.items():
        editor = ParamEditor(on_change)
        editors.append(editor)

        tab = ttk.Frame(book)
        book.add(tab, text=fruit.title())

        area = ScrollableFrame(tab)
        area.pack(fill="both", expand=True)
        body = area.body
        body.columnconfigure(0, weight=1)

        row = 0
        for title, params in _sections(fruit, module):
            editor.section(body, title, params, row)
            row += 1

        ttk.Button(body, text=f"Reset {fruit} defaults",
                   command=editor.reset).grid(row=row, column=0,
                                              sticky="w", padx=4, pady=6)

    return _AllFruits(editors)
