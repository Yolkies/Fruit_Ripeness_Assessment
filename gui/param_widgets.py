"""Shared parameter-editing widgets used by the three module panels."""

import tkinter as tk
from tkinter import ttk

import numpy as np


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
    """One editable value on a target object (a params dataclass or a module)."""

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


def hsv_triplet(label, target, name, kind="int"):
    """Three Params for an [H, S, V] array or tuple constant."""
    limits = [(0, 179), (0, 255), (0, 255)]
    return [
        Param(f"{label} {axis}", target, name, (i,), kind, lo, hi, 1)
        for i, (axis, (lo, hi)) in enumerate(zip("HSV", limits))
    ]


def hsv_range_pair(label, target, name, path=()):
    """Six Params for a ((h,s,v), (h,s,v)) lower/upper range constant."""
    limits = [(0, 179), (0, 255), (0, 255)]
    params = []
    for bound, edge in enumerate(("lo", "hi")):
        for i, (axis, (lo, hi)) in enumerate(zip("HSV", limits)):
            params.append(Param(f"{label} {edge} {axis}", target, name,
                                path + (bound, i), "int", lo, hi, 1))
    return params


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, height=520, width=340):
        super().__init__(parent)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                           height=height, width=width)
        bar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.body = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>",
                       lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(window, width=e.width))

        def wheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))


class ParamEditor:
    """Renders Params as labelled rows and keeps the widgets in sync."""

    def __init__(self, on_change=None):
        self.on_change = on_change or (lambda: None)
        self.rows = []

    def section(self, parent, title, params, row):
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        frame.columnconfigure(1, weight=1)
        for i, param in enumerate(params):
            self.add(frame, param, i)
        return frame

    def add(self, parent, param, row):
        ttk.Label(parent, text=param.label).grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=1)

        if param.kind == "bool":
            var = tk.BooleanVar(value=bool(param.get()))

            def apply(*_):
                param.set(var.get())
                self.on_change()

            ttk.Checkbutton(parent, variable=var, command=apply).grid(
                row=row, column=1, columnspan=2, sticky="w")
            self.rows.append((param, var, None))
            return

        if param.kind == "text":
            var = tk.StringVar(value=str(param.get()))

            def apply(*_):
                param.set(var.get())
                self.on_change()

            entry = ttk.Entry(parent, textvariable=var)
            entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=2)
            entry.bind("<Return>", apply)
            entry.bind("<FocusOut>", apply)
            self.rows.append((param, var, None))
            return

        var = tk.DoubleVar(value=float(param.get()))
        text = tk.StringVar(value=param.format())
        guard = {"busy": False}

        def push(value):
            guard["busy"] = True
            applied = param.set(value)
            var.set(float(applied))
            text.set(param.format(applied))
            guard["busy"] = False
            self.on_change()

        def from_scale(_=None):
            if guard["busy"]:
                return
            push(var.get())

        def from_entry(_=None):
            if guard["busy"]:
                return
            try:
                push(float(text.get()))
            except ValueError:
                text.set(param.format())

        scale = ttk.Scale(parent, from_=param.lo, to=param.hi,
                          variable=var, command=from_scale)
        scale.grid(row=row, column=1, sticky="ew", padx=2)
        entry = ttk.Entry(parent, textvariable=text, width=7)
        entry.grid(row=row, column=2, sticky="w")
        entry.bind("<Return>", from_entry)
        entry.bind("<FocusOut>", from_entry)
        self.rows.append((param, var, text))

    def refresh(self):
        for param, var, text in self.rows:
            value = param.get()
            if isinstance(var, tk.DoubleVar):
                var.set(float(value))
            else:
                var.set(value)
            if text is not None:
                text.set(param.format(value))

    def reset(self):
        for param, _, _ in self.rows:
            param.set(param.default)
        self.refresh()
        self.on_change()
