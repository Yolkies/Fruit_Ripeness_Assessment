from tkinter import ttk

import ripeness
from gui.param_widgets import ParamEditor, ScrollableFrame
from params_spec import ripeness_sections


class _AllFruits:
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

    for fruit in ripeness.FRUIT_MODULES:
        editor = ParamEditor(on_change)
        editors.append(editor)

        tab = ttk.Frame(book)
        book.add(tab, text=fruit.title())

        area = ScrollableFrame(tab)
        area.pack(fill="both", expand=True)
        body = area.body
        body.columnconfigure(0, weight=1)

        row = 0
        for title, params in ripeness_sections(fruit):
            editor.section(body, title, params, row)
            row += 1

        ttk.Button(body, text=f"Reset {fruit} defaults", command=editor.reset).grid(
            row=row, column=0, sticky="w", padx=4, pady=6
        )

    return _AllFruits(editors)
