from tkinter import ttk

from gui.param_widgets import ParamEditor, ScrollableFrame
from params_spec import blemish_sections


def build(parent, on_change):
    editor = ParamEditor(on_change)

    area = ScrollableFrame(parent)
    area.pack(fill="both", expand=True)
    body = area.body
    body.columnconfigure(0, weight=1)

    row = 0
    for title, params in blemish_sections():
        editor.section(body, title, params, row)
        row += 1

    ttk.Button(body, text="Reset blemish defaults", command=editor.reset).grid(
        row=row, column=0, sticky="w", padx=4, pady=6
    )
    return editor
