import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, height=520, width=340):
        super().__init__(parent)
        canvas = tk.Canvas(
            self, borderwidth=0, highlightthickness=0, height=height, width=width
        )
        bar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.body = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfigure(window, width=e.width)
        )

        def wheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))


class ParamEditor:
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
            row=row, column=0, sticky="w", padx=(0, 6), pady=1
        )

        if param.kind == "bool":
            var = tk.BooleanVar(value=bool(param.get()))

            def apply(*_):
                param.set(var.get())
                self.on_change()

            ttk.Checkbutton(parent, variable=var, command=apply).grid(
                row=row, column=1, columnspan=2, sticky="w"
            )
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

        scale = ttk.Scale(
            parent, from_=param.lo, to=param.hi, variable=var, command=from_scale
        )
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
