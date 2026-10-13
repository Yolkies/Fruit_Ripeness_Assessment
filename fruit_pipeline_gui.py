import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

import blemish_detection
import detection
import preprocessing
import ripeness
import training
from detection import (
    CLASS_NAMES,
    annotate,
    contour_count,
    crop_detection_from_original,
    export_for_ripeness,
    load_model,
    segment,
)
from gui import (
    blemish_panel,
    detection_panel,
    preprocessing_panel,
    ripeness_panel,
    ripeness_result,
)
from gui.param_widgets import ParamEditor, ScrollableFrame
from params_spec import Param

DISPLAY_MAX = 480


@dataclass
class AppParams:
    camera_index: int = 0
    panel_size: int = 380
    crops_per_page: int = 4
    zoom_pad_percent: float = 0.15
    auto_rerun: bool = False
    skip_fruit_confirm: bool = False


APP = AppParams()


def build_crop_grid(crops, panel):
    cols = 1 if len(crops) == 1 else 2
    rows = 1 if len(crops) <= 2 else 2
    tw, th = panel // cols, panel // rows
    grid = np.full((th * rows, tw * cols, 3), 238, dtype=np.uint8)
    for i, crop in enumerate(crops):
        h, w = crop.shape[:2]
        s = min(tw / w, th / h)
        rw, rh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(crop, (rw, rh))
        r, col = divmod(i, cols)
        y0 = r * th + (th - rh) // 2
        x0 = col * tw + (tw - rw) // 2
        grid[y0 : y0 + rh, x0 : x0 + rw] = small
    return grid


class PipelineGUI:
    def __init__(self, root):
        self.root = root
        root.title("Fruit Segmentation Pipeline")
        self.PANEL = APP.panel_size
        self.model = None
        self.current_img = None
        self.pre_img = None
        self.seg_page = 0
        self.detections = []
        self.ripeness_results = {}
        self.blemish_results = []
        self.fruit_overrides = {}
        self.ripeness_current_index = 0
        self.show_mask = tk.BooleanVar(value=False)
        self.auto_rerun = tk.BooleanVar(value=APP.auto_rerun)
        self._rerun_job = None
        self._train_queue = queue.Queue()
        self._training = False

        bar = ttk.Frame(root, padding=8)
        bar.pack(fill="x")
        ttk.Button(bar, text="Load Image", command=self.load_image).pack(
            side="left", padx=4
        )
        ttk.Button(bar, text="Capture Camera", command=self.capture_camera).pack(
            side="left", padx=4
        )
        ttk.Button(bar, text="Run Pipeline", command=self.run_pipeline).pack(
            side="left", padx=4
        )
        ttk.Button(bar, text="Run Ripeness", command=self.run_ripeness).pack(
            side="left", padx=4
        )
        ttk.Button(bar, text="Run Blemish", command=self.run_blemish).pack(
            side="left", padx=4
        )
        ttk.Button(bar, text="Blemish PDF", command=self.export_blemish_pdf).pack(
            side="left", padx=4
        )
        ttk.Button(bar, text="Export for Ripeness", command=self.export).pack(
            side="left", padx=4
        )
        self.train_btn = ttk.Button(
            bar, text="Train Model", command=self.start_training
        )
        self.train_btn.pack(side="left", padx=4)
        self.include_negatives = tk.BooleanVar(value=training.PARAMS.include_negatives)
        ttk.Checkbutton(
            bar, text="Include false images", variable=self.include_negatives
        ).pack(side="left", padx=4)
        ttk.Checkbutton(
            bar,
            text="Show mask overlay",
            variable=self.show_mask,
            command=self.refresh_segmented,
        ).pack(side="left", padx=12)

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        panels = ttk.Frame(left, padding=8)
        panels.pack(fill="both", expand=False)
        self.panel = {}
        self._placeholder = ImageTk.PhotoImage(
            Image.new("RGB", (self.PANEL, self.PANEL), (238, 238, 238))
        )
        for col, (key, title) in enumerate(
            [
                ("orig", "1. Original"),
                ("pre", "2. Preprocessed"),
                ("seg", "3. Segmented + Contours"),
            ]
        ):
            frame = ttk.LabelFrame(panels, text=title, padding=4)
            frame.grid(row=0, column=col, padx=6, pady=6, sticky="n")
            holder = tk.Frame(frame, width=self.PANEL, height=self.PANEL)
            holder.pack()
            holder.pack_propagate(False)
            lbl = tk.Label(holder, bg="#eeeeee", image=self._placeholder)
            lbl.pack(fill="both", expand=True)
            self.panel[key] = lbl

            if key == "pre":
                nav = ttk.Frame(frame)
                nav.pack(fill="x", pady=(4, 0))
                self.crop_label = ttk.Label(nav, text="")
                self.crop_label.pack(side="top")

            if key == "seg":
                nav = ttk.Frame(frame)
                nav.pack(fill="x", pady=(4, 0))
                self.seg_prev_btn = ttk.Button(
                    nav,
                    text="<",
                    width=3,
                    state="disabled",
                    command=lambda: self.seg_page_step(-1),
                )
                self.seg_prev_btn.pack(side="left")
                self.seg_next_btn = ttk.Button(
                    nav,
                    text=">",
                    width=3,
                    state="disabled",
                    command=lambda: self.seg_page_step(1),
                )
                self.seg_next_btn.pack(side="right")
                self.seg_label = ttk.Label(nav, text="")
                self.seg_label.pack(side="top")

        self.info = tk.Text(left, height=8)
        self.info.pack(fill="both", expand=True, padx=8, pady=6)

        self.build_parameter_panels(body)

        self.log(f"Model: {detection.PARAMS.model_path}")
        self.log("Load an image (or capture from camera), then Run Pipeline.")

    def build_parameter_panels(self, parent):
        side = ttk.Frame(parent, padding=(0, 8, 8, 8))
        side.pack(side="right", fill="both", expand=False)

        head = ttk.Frame(side)
        head.pack(fill="x", pady=(0, 4))
        ttk.Label(head, text="Parameters").pack(side="left")
        ttk.Checkbutton(
            head,
            text="Auto re-run",
            variable=self.auto_rerun,
            command=self.on_param_change,
        ).pack(side="right")

        book = ttk.Notebook(side)
        book.pack(fill="both", expand=True)

        self.editors = {}
        for key, title, panel in [
            ("2.1", "2.1 Pre-processing", preprocessing_panel),
            ("2.2", "2.2 Detection", detection_panel),
            ("2.3", "2.3 Ripeness", ripeness_panel),
            ("2.4", "2.4 Blemish", blemish_panel),
        ]:
            tab = ttk.Frame(book, padding=4)
            book.add(tab, text=title)
            if key == "2.2":
                self.editors[key] = panel.build(
                    tab, self.on_param_change, on_model_change=self.reload_model
                )
            else:
                self.editors[key] = panel.build(tab, self.on_param_change)

        app_tab = ttk.Frame(book, padding=4)
        book.add(app_tab, text="App")
        area = ScrollableFrame(app_tab)
        area.pack(fill="both", expand=True)
        area.body.columnconfigure(0, weight=1)
        app_editor = ParamEditor(lambda: None)
        app_editor.section(
            area.body,
            "Application",
            [
                Param("Camera index", APP, "camera_index", kind="int", lo=0, hi=8),
                Param("Crops per page", APP, "crops_per_page", kind="int", lo=1, hi=4),
                Param(
                    "Single-fruit zoom pad",
                    APP,
                    "zoom_pad_percent",
                    kind="float",
                    lo=0.0,
                    hi=1.0,
                ),
                Param(
                    "Skip fruit confirm (blemish)",
                    APP,
                    "skip_fruit_confirm",
                    kind="bool",
                ),
            ],
            0,
        )
        self.editors["app"] = app_editor

    def on_param_change(self):
        if self.current_img is None:
            return
        self.refresh_segmented()
        if not self.auto_rerun.get():
            return
        if self._rerun_job is not None:
            self.root.after_cancel(self._rerun_job)
        self._rerun_job = self.root.after(400, self.rerun)

    def rerun(self):
        self._rerun_job = None
        if self.current_img is None:
            return
        if not self.ensure_model():
            return
        try:
            self.process_frame(self.current_img, log_result=False)
        except Exception as exc:
            self.log(f"Pipeline failed with the current parameters: {exc}")

    def reload_model(self):
        self.model = None
        self.log(f"Model path set to: {detection.PARAMS.model_path}")

    def log(self, msg):
        self.info.insert("end", msg + "\n")
        self.info.see("end")

    def ensure_model(self):
        if self.model is None:
            path = detection.resolve_path(detection.PARAMS.model_path)
            if not os.path.exists(path):
                messagebox.showerror(
                    "Model not found",
                    f"Could not find the model at:\n{path}\n"
                    "Set the weights path on the 2.2 Detection tab.",
                )
                return False
            self.log("Loading model...")
            self.model = load_model()
            self.log("Model loaded.")
        return True

    def show(self, key, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = self.PANEL / max(h, w)
        rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
        canvas = np.full((self.PANEL, self.PANEL, 3), 238, dtype=np.uint8)
        yh, xw = rgb.shape[:2]
        y0 = (self.PANEL - yh) // 2
        x0 = (self.PANEL - xw) // 2
        canvas[y0 : y0 + yh, x0 : x0 + xw] = rgb
        imgtk = ImageTk.PhotoImage(Image.fromarray(canvas))
        lbl = self.panel[key]
        lbl.configure(image=imgtk)
        lbl.image = imgtk

    def clear_panel(self, key):
        self.panel[key].configure(image=self._placeholder)
        self.panel[key].image = self._placeholder

    def show_preprocessed(self):
        if self.pre_img is None:
            self.clear_panel("pre")
            self.crop_label.configure(text="")
            return
        self.show("pre", self.pre_img)
        h, w = self.pre_img.shape[:2]
        self.crop_label.configure(text=f"pre-processed  |  {w}x{h} px")

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        self.current_img = cv2.imread(path)
        self.pre_img = None
        self.detections = []
        self.reset_per_image_state()
        self.show("orig", self.current_img)
        self.show_preprocessed()
        self.clear_panel("seg")
        self.log(f"Loaded: {os.path.basename(path)}")

    def capture_camera(self):
        cap = cv2.VideoCapture(APP.camera_index)
        if not cap.isOpened():
            messagebox.showerror("Camera", "Could not open camera.")
            return
        self.log("Camera live. Click 'Capture frame' or press SPACE; Esc cancels.")

        win = tk.Toplevel(self.root)
        win.title("Camera - SPACE/Capture to grab, Esc to cancel")
        view = tk.Label(win)
        view.pack()
        state = {"frame": None, "run": True}

        def grab():
            if not state["run"]:
                return
            ok, f = cap.read()
            if ok:
                state["frame"] = f
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
                view.configure(image=imgtk)
                view.image = imgtk
            win.after(30, grab)

        def capture(_=None):
            if state["frame"] is not None:
                self.current_img = state["frame"].copy()
                self.pre_img = None
                self.detections = []
                self.reset_per_image_state()
                self.show("orig", self.current_img)
                self.show_preprocessed()
                self.clear_panel("seg")
                self.log("Captured frame from camera.")
            close()

        def close(_=None):
            state["run"] = False
            cap.release()
            win.destroy()

        ttk.Button(win, text="Capture frame", command=capture).pack(pady=4)
        win.bind("<space>", capture)
        win.bind("<Escape>", close)
        win.protocol("WM_DELETE_WINDOW", close)
        grab()

    def process_frame(self, img_bgr, log_result=True):
        self.current_img = img_bgr

        prep_result = preprocessing.preprocess(img_bgr)
        self.pre_img = prep_result.bgr

        self.seg_page = 0

        self.detections = segment(self.model, self.pre_img, img_bgr.shape)

        count = contour_count(self.detections)

        self.show("orig", self.current_img)
        self.show_preprocessed()
        self.refresh_segmented()

        per_class = {}
        for d in self.detections:
            per_class[CLASS_NAMES[d["cls"]]] = (
                per_class.get(CLASS_NAMES[d["cls"]], 0) + 1
            )
        if log_result:
            self.log(f"\nContour count (fruits detected): {count}")
            if per_class:
                breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(per_class.items()))
                self.log(f"Fruit summary -> {breakdown}")
                self.log("Individual detections:")
                for i, d in enumerate(self.detections):
                    fruit_name = CLASS_NAMES[d["cls"]]
                    confidence = d["conf"]
                    self.log(
                        f"  Fruit {i+1}: {fruit_name} (Confidence: {confidence:.2f})"
                    )

            else:
                self.log("No fruit detected (image may be a negative/background).")
        return count, per_class

    def run_pipeline(self):
        if self.current_img is None:
            messagebox.showwarning(
                "No image", "Load an image or capture from camera first."
            )
            return
        if not self.ensure_model():
            return
        try:
            self.process_frame(self.current_img)
        except Exception as exc:
            self.log(f"Pipeline failed with the current parameters: {exc}")

    def refresh_segmented(self):
        if self.current_img is None:
            return
        base = self.pre_img if self.pre_img is not None else self.current_img
        if not self.detections:
            self.show("seg", base)
            self.seg_label.configure(text="")
            self.seg_prev_btn.configure(state="disabled")
            self.seg_next_btn.configure(state="disabled")
            return
        pages = len(self.detections) + 1
        self.seg_page = max(0, min(self.seg_page, pages - 1))
        shown = (
            self.detections
            if self.seg_page == 0
            else [self.detections[self.seg_page - 1]]
        )
        annotated = annotate(base, shown, show_mask=self.show_mask.get())
        if self.seg_page > 0:
            x, y, w, h = cv2.boundingRect(shown[0]["mask"])
            if w and h:
                pad = int(max(w, h) * APP.zoom_pad_percent)
                H, W = annotated.shape[:2]
                annotated = annotated[
                    max(0, y - pad) : min(H, y + h + pad),
                    max(0, x - pad) : min(W, x + w + pad),
                ]
        self.show("seg", annotated)
        self.seg_label.configure(
            text=(
                f"{len(self.detections)} fruit(s)  |  all"
                if self.seg_page == 0
                else f"fruit {self.seg_page}/{len(self.detections)}"
            )
        )
        self.seg_prev_btn.configure(state="normal")
        self.seg_next_btn.configure(state="normal")

    def seg_page_step(self, step):
        if not self.detections:
            return
        self.seg_page = (self.seg_page + step) % (len(self.detections) + 1)
        self.refresh_segmented()

    def start_training(self):
        if self._training:
            return
        training.PARAMS.include_negatives = self.include_negatives.get()
        run_name = training.run_name_for()
        if not messagebox.askyesno(
            "Train model",
            "Build the dataset (split into train/val) and start YOLO training?\n\n"
            f"Source: {training.PARAMS.source_dir}\n"
            f"Dataset: {training.PARAMS.dataset_dir}\n"
            f"Run: {training.PARAMS.runs_dir}/{run_name}\n"
            f"False images: {'included' if training.PARAMS.include_negatives else 'excluded'}\n"
            f"Epochs: {training.PARAMS.epochs}\n"
            f"Patience: {training.PARAMS.patience}\n\n"
            "This overwrites the dataset folder and the run folder "
            "(including its current best.pt), and can take a long time.",
        ):
            return

        self._training = True
        self.train_btn.configure(state="disabled", text="Training...")
        self.log("\nStarting training pipeline.")

        weights = f"{training.PARAMS.runs_dir}/{run_name}/weights/best.pt"

        def worker():
            try:
                command = [
                    sys.executable,
                    "-u",
                    "training.py",
                    "--negatives",
                    "1" if training.PARAMS.include_negatives else "0",
                    "--epochs",
                    str(training.PARAMS.epochs),
                    "--patience",
                    str(training.PARAMS.patience),
                    "--batch",
                    str(training.PARAMS.batch),
                    "--workers",
                    str(training.PARAMS.workers),
                    "--imgsz",
                    str(training.PARAMS.imgsz),
                    "--device",
                    str(training.PARAMS.device),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=str(training.PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        self._train_queue.put(line)
                code = process.wait()
                if code == 0:
                    self._train_queue.put(("done", weights))
                else:
                    self._train_queue.put(
                        ("failed", f"training.py exited with code {code}")
                    )
            except Exception as exc:
                self._train_queue.put(("failed", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(200, self._drain_training)

    def _drain_training(self):
        while True:
            try:
                item = self._train_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                status, detail = item
                self._training = False
                self.train_btn.configure(state="normal", text="Train Model")
                if status == "done":
                    self.log(f"Training complete: {detail}")
                    if os.path.exists(detection.resolve_path(detail)):
                        detection.PARAMS.model_path = detail
                        self.editors["2.2"].refresh()
                        self.model = None
                        self.log("Detection now uses the newly trained weights.")
                else:
                    self.log(f"Training failed: {detail}")
                return
            self.log(str(item))
        if self._training:
            self.root.after(200, self._drain_training)

    def reset_per_image_state(self):
        self.fruit_overrides.clear()
        self.blemish_results = []
        self.ripeness_results = {}
        self.ripeness_current_index = 0

    def confirm_fruit_types(self):
        popup = tk.Toplevel(self.root)
        popup.title("Confirm Fruit Types")
        popup.transient(self.root)
        popup.image_refs = []

        ttk.Label(
            popup,
            text="Confirm the fruit type for each detection",
            font=("Arial", 11, "bold"),
        ).pack(pady=(12, 2))
        ttk.Label(
            popup,
            text="The label is a caption for the blemish report; "
            "it does not change the defect measurement.",
            foreground="#666666",
        ).pack(pady=(0, 8))

        area = ScrollableFrame(popup)
        area.pack(fill="both", expand=True, padx=12)

        options = ["Apple", "Banana", "Mango", "Orange", "Strawberry", "Tomato"]
        rows = []
        for index, d in enumerate(self.detections):
            yolo_type = CLASS_NAMES[d["cls"]]
            line = ttk.Frame(area.body)
            line.pack(fill="x", pady=3)

            crop, _mask, _masked = crop_detection_from_original(self.current_img, d)
            thumb = ripeness_result.cv_to_tk_image(crop, max_width=56, max_height=56)
            if thumb is not None:
                popup.image_refs.append(thumb)
                ttk.Label(line, image=thumb).pack(side="left", padx=(0, 8))

            ttk.Label(line, text=f"Fruit {index + 1}", width=8).pack(side="left")
            ttk.Label(
                line,
                text=f"YOLO: {yolo_type.title()} ({d['conf'] * 100:.0f}%)",
                width=22,
                foreground="#666666",
            ).pack(side="left")

            current = self.fruit_overrides.get(index, yolo_type)
            var = tk.StringVar(value=current.title())
            ttk.Combobox(
                line, textvariable=var, values=options, state="readonly", width=12
            ).pack(side="left", padx=4)
            rows.append((index, yolo_type, var))

        state = {"proceed": False}

        def apply_and_run():
            for index, yolo_type, var in rows:
                chosen = var.get().lower()
                if chosen != yolo_type:
                    self.fruit_overrides[index] = chosen
                else:
                    self.fruit_overrides.pop(index, None)
            state["proceed"] = True
            popup.destroy()

        def use_yolo():
            state["proceed"] = True
            popup.destroy()

        buttons = ttk.Frame(popup)
        buttons.pack(fill="x", padx=12, pady=12)
        ttk.Button(buttons, text="Apply and run blemish", command=apply_and_run).pack(
            side="left", expand=True, fill="x", padx=4
        )
        ttk.Button(buttons, text="Use YOLO labels", command=use_yolo).pack(
            side="left", expand=True, fill="x", padx=4
        )
        ttk.Button(buttons, text="Cancel", command=popup.destroy).pack(
            side="left", expand=True, fill="x", padx=4
        )

        popup.update_idletasks()
        popup.grab_set()
        self.root.wait_window(popup)
        return state["proceed"]

    def run_blemish(self, confirm=True):
        if not self.detections:
            messagebox.showwarning("No detections", "Run the pipeline first.")
            return

        if confirm and not APP.skip_fruit_confirm:
            if not self.confirm_fruit_types():
                return

        self.log("\nBlemish detection (Siridhara et al. 2023):")
        self.blemish_results = []
        overlays = []

        for i, d in enumerate(self.detections, start=1):
            crop, mask, masked_crop = crop_detection_from_original(self.current_img, d)
            yolo_type = CLASS_NAMES[d["cls"]]
            fruit_type = self.fruit_overrides.get(i - 1, yolo_type)
            try:
                result, diagnostics = blemish_detection.assess_single_fruit(crop, mask)
            except ValueError as exc:
                self.log(f"  Fruit {i} ({fruit_type}): skipped - {exc}")
                continue

            result["number"] = i
            result["fruit_type"] = fruit_type
            result["yolo_fruit_type"] = yolo_type
            result["fruit_type_overridden"] = fruit_type != yolo_type
            result["diagnostics"] = diagnostics
            self.blemish_results.append(result)
            overlays.append(diagnostics["06_overlay"])

            label = fruit_type
            if result["fruit_type_overridden"]:
                label += f" (corrected from {yolo_type})"
            self.log(
                f"  Fruit {i} ({label}): {result['status']}"
                f"  {result['defect_area_pct']:.2f}% of surface"
                f"  ({result['blob_count']} region(s))"
            )

        if not self.blemish_results:
            self.log("  No fruit could be assessed.")
            return

        defected = sum(1 for r in self.blemish_results if r["status"] == "Defected")
        self.log(
            f"  Summary: {defected} defected, "
            f"{len(self.blemish_results) - defected} good."
        )

        self.show("pre", build_crop_grid(overlays[: APP.crops_per_page], self.PANEL))
        self.crop_label.configure(
            text=f"blemish overlays  |  {defected} defected / "
            f"{len(self.blemish_results)}"
        )

    def export_blemish_pdf(self):
        if not self.blemish_results:
            messagebox.showwarning("No blemish results", "Run Blemish first.")
            return
        try:
            import blemish_report
        except ImportError:
            messagebox.showerror(
                "reportlab missing",
                "PDF export needs reportlab.\n\nInstall it with:\n"
                "    pip install reportlab",
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile="blemish_report.pdf",
            filetypes=[("PDF", "*.pdf")],
        )
        if not path:
            return

        try:
            annotated = annotate(self.current_img, self.detections, show_mask=True)
            pdf = blemish_report.build_pdf(
                self.blemish_results, annotated_image=annotated
            )
            with open(path, "wb") as handle:
                handle.write(pdf)
        except Exception as exc:
            messagebox.showerror("PDF export failed", str(exc))
            return

        self.log(f"\nBlemish PDF saved: {os.path.abspath(path)}")

    def run_ripeness(self):
        if not self.detections:
            messagebox.showwarning("No detections", "Run the pipeline first.")
            return

        self.ripeness_current_index = 0
        self.show_ripeness_confirmation()

    def prepare_ripeness_detection(self, index):
        d = self.detections[index]

        crop, mask, masked_crop = crop_detection_from_original(self.current_img, d)

        return {
            "number": index + 1,
            "yolo_fruit_type": CLASS_NAMES[d["cls"]],
            "confidence": float(d["conf"]),
            "crop": crop,
            "fruit_mask": mask,
            "masked_crop": masked_crop,
        }

    def show_ripeness_confirmation(self):
        detected_fruit = self.prepare_ripeness_detection(self.ripeness_current_index)

        predicted_fruit = detected_fruit["yolo_fruit_type"]
        confidence = detected_fruit["confidence"]
        crop = detected_fruit.get("crop")

        popup = tk.Toplevel(self.root)
        popup.title("Confirm Fruit Type")
        popup.geometry("520x620")
        popup.transient(self.root)

        popup.image_refs = []

        ttk.Label(
            popup,
            text=f"Fruit {self.ripeness_current_index + 1} of {len(self.detections)}",
            font=("Arial", 14, "bold"),
        ).pack(pady=(20, 10))
        ttk.Label(
            popup,
            text=f"YOLO Prediction: {predicted_fruit.title()}",
            font=("Arial", 12, "bold"),
        ).pack(pady=5)
        ttk.Label(popup, text=f"Confidence: {confidence * 100:.1f}%").pack(pady=5)

        if crop is not None:
            ttk.Label(popup, text="Detected Fruit", font=("Arial", 11, "bold")).pack(
                pady=(12, 5)
            )

            crop_tk = ripeness_result.cv_to_tk_image(
                crop, max_width=260, max_height=220
            )

            if crop_tk is not None:
                popup.image_refs.append(crop_tk)
                ttk.Label(popup, image=crop_tk).pack(pady=5)

        ttk.Label(popup, text="Is the detected fruit type correct?").pack(pady=(15, 5))

        fruit_options = ["Apple", "Banana", "Mango", "Orange", "Strawberry", "Tomato"]

        selected_fruit_var = tk.StringVar(value=predicted_fruit.title())

        ttk.Combobox(
            popup,
            textvariable=selected_fruit_var,
            values=fruit_options,
            state="readonly",
        ).pack(pady=10)

        def continue_with_yolo():
            try:
                result = ripeness.assess_detected_fruit(detected_fruit)

                self.ripeness_results[self.ripeness_current_index] = result

                popup.destroy()

                ripeness_result.show_ripeness_result(self.root, result)

            except Exception as exc:
                messagebox.showerror("Ripeness Assessment Error", str(exc))

        def analyse_selected():
            selected_fruit = selected_fruit_var.get()

            try:
                result = ripeness.assess_detected_fruit(
                    detected_fruit, selected_fruit_type=selected_fruit
                )

                self.fruit_overrides[self.ripeness_current_index] = (
                    selected_fruit.lower()
                )

                self.ripeness_results[self.ripeness_current_index] = result

                popup.destroy()

                ripeness_result.show_ripeness_result(self.root, result)

            except Exception as exc:
                messagebox.showerror("Ripeness Assessment Error", str(exc))

        button_frame = ttk.Frame(popup)
        button_frame.pack(fill="x", padx=20, pady=15)

        ttk.Button(
            button_frame,
            text=f"Continue with {predicted_fruit.title()}",
            command=continue_with_yolo,
        ).pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(
            button_frame, text="Analyse as Selected Fruit", command=analyse_selected
        ).pack(side="left", expand=True, fill="x", padx=5)

        nav_frame = ttk.Frame(popup)
        nav_frame.pack(fill="x", padx=20, pady=(10, 5))

        def previous_fruit():
            popup.destroy()
            self.ripeness_current_index = (self.ripeness_current_index - 1) % len(
                self.detections
            )
            self.show_ripeness_confirmation()

        def next_fruit():
            popup.destroy()
            self.ripeness_current_index = (self.ripeness_current_index + 1) % len(
                self.detections
            )
            self.show_ripeness_confirmation()

        ttk.Button(nav_frame, text="< Previous Fruit", command=previous_fruit).pack(
            side="left", padx=5
        )
        ttk.Button(nav_frame, text="Next Fruit >", command=next_fruit).pack(
            side="right", padx=5
        )

    def export(self):
        if not self.detections:
            messagebox.showwarning("Nothing to export", "Run the pipeline first.")
            return
        session, manifest = export_for_ripeness(self.current_img, self.detections)
        self.log(
            f"\nExported {len(manifest)} masked fruit crop(s) for the ripeness stage:"
        )
        self.log(f"  {os.path.abspath(session)}")
        self.log("  (manifest.json lists each fruit's type, area, and crop file.)")

    def on_close(self):
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
