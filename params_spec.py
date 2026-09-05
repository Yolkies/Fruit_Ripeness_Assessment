import numpy as np

import detection
import preprocessing
import ripeness


def _replace(container, path, value):
    key, rest = path[0], path[1:]
    inner = value if not rest else _replace(container[key], rest, value)
    if isinstance(container, np.ndarray):
        updated = container.copy()
        updated[key] = inner
        return updated
    items = list(container)
    items[key] = inner
    return tuple(items)


class Param:
    def __init__(self, label, target, name, path=(), kind="float",
                 lo=0.0, hi=1.0, step=0.01):
        self.label = label
        self.target = target
        self.name = name
        self.path = tuple(path)
        self.kind = kind
        self.lo = lo
        self.hi = hi
        self.step = step
        self.default = self.get()

    def get(self):
        value = getattr(self.target, self.name)
        for key in self.path:
            value = value[key]
        return value

    def set(self, value):
        if self.kind == "int":
            value = int(round(float(value)))
        elif self.kind == "float":
            value = float(value)
        elif self.kind == "bool":
            value = bool(value)
        if self.kind in ("int", "float"):
            value = max(self.lo, min(self.hi, value))
        if not self.path:
            setattr(self.target, self.name, value)
        else:
            setattr(self.target, self.name,
                    _replace(getattr(self.target, self.name), self.path, value))
        return value

    def format(self, value=None):
        value = self.get() if value is None else value
        if self.kind == "int":
            return str(int(value))
        if self.kind == "float":
            return f"{float(value):g}"
        return str(value)

    def key(self):
        return f"{self.target.__class__.__name__}.{self.name}.{self.path}"


def hsv_triplet(label, target, name, kind="int"):
    limits = [(0, 179), (0, 255), (0, 255)]
    return [
        Param(f"{label} {axis}", target, name, (i,), kind, lo, hi, 1)
        for i, (axis, (lo, hi)) in enumerate(zip("HSV", limits))
    ]


def hsv_range_pair(label, target, name, path=()):
    limits = [(0, 179), (0, 255), (0, 255)]
    params = []
    for bound, edge in enumerate(("lo", "hi")):
        for i, (axis, (lo, hi)) in enumerate(zip("HSV", limits)):
            params.append(Param(f"{label} {edge} {axis}", target, name,
                                path + (bound, i), "int", lo, hi, 1))
    return params


def _hue_range(label, module, name, path=()):
    return [
        Param(f"{label} hue lo", module, name, path + (0,), "int", 0, 179, 1),
        Param(f"{label} hue hi", module, name, path + (1,), "int", 0, 179, 1),
    ]


def _pct(label, module, name):
    return Param(label, module, name, kind="float", lo=0.0, hi=100.0, step=0.5)


def preprocessing_sections():
    p = preprocessing.PARAMS
    return [
        ("1.1 Resize / normalise", [
            Param("Enabled", p, "resize_enabled", kind="bool"),
            Param("Target long side", p, "target_long_side", kind="int", lo=128, hi=2048),
        ]),
        ("1.2 White balance", [
            Param("Enabled", p, "white_balance_enabled", kind="bool"),
            Param("Saturation thr.", p, "wb_saturation_threshold", kind="float", lo=0.0, hi=1.0),
            Param("Strength", p, "wb_strength", kind="float", lo=0.0, hi=1.0),
            Param("Max gain", p, "wb_max_gain", kind="float", lo=1.0, hi=4.0),
        ]),
        ("1.3 Denoise", [
            Param("Enabled", p, "denoise_enabled", kind="bool"),
            Param("Median ksize (odd)", p, "median_ksize", kind="int", lo=1, hi=15),
            Param("Bilateral d", p, "bilateral_d", kind="int", lo=1, hi=15),
            Param("Sigma colour", p, "bilateral_sigma_colour", kind="int", lo=1, hi=200),
            Param("Sigma space", p, "bilateral_sigma_space", kind="int", lo=1, hi=200),
        ]),
        ("1.4 CLAHE", [
            Param("Enabled", p, "clahe_enabled", kind="bool"),
            Param("Clip limit", p, "clahe_clip_limit", kind="float", lo=0.1, hi=10.0),
            Param("Tile grid", p, "clahe_tile_grid", kind="int", lo=1, hi=32),
            Param("Contrast gate", p, "clahe_contrast_gate", kind="float", lo=0.0, hi=128.0),
        ]),
        ("1.5 Glare removal", [
            Param("Enabled", p, "glare_enabled", kind="bool"),
            Param("V min", p, "glare_v_min", kind="int", lo=0, hi=255),
            Param("S max", p, "glare_s_max", kind="int", lo=0, hi=255),
            Param("Inpaint radius", p, "glare_inpaint_radius", kind="int", lo=1, hi=25),
        ]),
        ("1.6 Otsu cropping", [
            Param("Min area (px)", p, "otsu_min_area", kind="int", lo=0, hi=50000),
            Param("Pad percent", p, "otsu_pad_percent", kind="float", lo=0.0, hi=1.0),
            Param("Blur ksize (odd)", p, "otsu_blur_ksize", kind="int", lo=1, hi=51),
            Param("Morph ksize", p, "otsu_morph_ksize", kind="int", lo=1, hi=51),
            Param("Dilate ksize", p, "otsu_dilate_ksize", kind="int", lo=1, hi=51),
        ]),
    ]


def detection_sections():
    p = detection.PARAMS
    return [
        ("Model", [
            Param("Weights", p, "model_path", kind="text"),
        ]),
        ("Segmentation", [
            Param("Confidence", p, "conf_threshold", kind="float", lo=0.01, hi=1.0),
            Param("Mask binary thr.", p, "mask_binary_threshold", kind="float", lo=0.01, hi=0.99),
            Param("Dedupe overlap", p, "dedupe_overlap_threshold", kind="float", lo=0.0, hi=1.0),
            Param("Retina masks", p, "retina_masks", kind="bool"),
        ]),
        ("Annotation", [
            Param("Overlay alpha", p, "overlay_alpha", kind="float", lo=0.0, hi=1.0),
            Param("Contour thickness", p, "contour_thickness", kind="int", lo=1, hi=10),
            Param("Label font scale", p, "label_font_scale", kind="float", lo=0.2, hi=3.0),
        ]),
        ("Export", [
            Param("Output folder", p, "output_dir", kind="text"),
        ]),
    ]


def ripeness_sections(fruit):
    module = ripeness.FRUIT_MODULES[fruit]

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
