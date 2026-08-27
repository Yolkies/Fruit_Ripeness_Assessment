"""
Fruit Segmentation Pipeline GUI
================================
Implements the front-end stages of the project flowchart:

    Input (image / camera)
        -> Preprocessing
        -> Fruit Segmentation        (YOLO26-seg instance masks)
        -> Contour Count             (classical cv2.findContours on each mask)
        -> Fruit Identification      (class label per instance)
        -> Output + hand-off         (masked fruit crops saved for the ripeness stage)

The ripeness / defect / grading stage is intentionally left as a stub
(`ripeness_stage`) for a teammate to implement. This GUI produces exactly what
that stage needs: one masked crop per fruit plus a manifest describing each.

Run inside the venv that has ultralytics + CUDA torch:
    python fruit_pipeline_gui.py
"""

import os
import json
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

import preprocessing


CLASS_NAMES = ["apple", "banana", "mango", "orange", "strawberry", "tomato"]
OUTPUT_DIR = "pipeline_output"          # masked crops for the ripeness stage go here
CONF_THRESHOLD = 0.2              # detection confidence cut-off
DISPLAY_MAX = 480                       # max px for on-screen preview

MODEL_PATH = r"best.pt"



def preprocess(img_bgr):
    """
    Flowchart: Preprocessing.
    Delegates to the Stage-1 preprocessing module (white balance, edge-preserving
    denoise, conditional CLAHE, glare removal, size normalisation). Returns the
    cleaned BGR image; segmentation and all later steps run on this.
    """
    result = preprocessing.preprocess(img_bgr)     # PreprocessResult
    return result.bgr


def segment(model, crops, crop_rects, orig_shape, conf=CONF_THRESHOLD):
    """
    Flowchart: Fruit Segmentation.
    Returns a list of per-instance dicts: binary mask, class id, confidence.
    """
    H, W = orig_shape[:2]
    detections = []

    for crop, (cx, cy, cw, ch) in zip(crops, crop_rects):
        res = model.predict(crop, conf=conf, retina_masks=True, verbose=False)[0]
        if res.masks is None:
            continue

        masks = res.masks.data.cpu().numpy()          # (N, ch, cw) in 0..1
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        for m, c, cf in zip(masks, cls_ids, confs):
            mask_resized = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 0.5).astype("uint8") * 255

            global_mask = np.zeros((H, W), dtype="uint8")
            global_mask[cy:cy+ch, cx:cx+cw] = mask_binary

            detections.append({"mask": global_mask, "cls": int(c), "conf": float(cf)})

    return dedupe(detections)


def dedupe(detections, overlap_thr=0.4):
    kept = []
    for d in sorted(detections, key=lambda x: -x["conf"]):
        area = cv2.countNonZero(d["mask"])
        if area == 0:
            continue
        if any(cv2.countNonZero(cv2.bitwise_and(d["mask"], k["mask"])) /
               min(area, cv2.countNonZero(k["mask"])) > overlap_thr for k in kept):
            continue
        kept.append(d)
    return sorted(kept, key=lambda d: cv2.boundingRect(d["mask"])[0])


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
        grid[y0:y0 + rh, x0:x0 + rw] = small
    return grid


def contour_count(detections):
    """
    Flowchart: Contour Count.
    Classical step: run cv2.findContours on EACH instance mask. Because the
    model separates touching fruit into individual masks, each mask yields one
    clean external contour -> an accurate count even when fruit touch (which
    plain thresholding + contours would merge into one).
    Adds 'contour' and 'area' to each detection; returns the total count.
    """
    count = 0
    for d in detections:
        cnts, _ = cv2.findContours(d["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            d["contour"] = None
            d["area"] = 0
            continue
        largest = max(cnts, key=cv2.contourArea)
        d["contour"] = largest
        d["area"] = float(cv2.contourArea(largest))
        count += 1
    return count


def annotate(img_bgr, detections, show_mask=False):
    """Draw contours + class labels + index for the Segmented view.
    If show_mask=True, also blend each instance's filled mask as a translucent
    coloured overlay."""
    out = img_bgr.copy()
    palette = [(66, 66, 244), (48, 172, 220), (60, 76, 231), (80, 175, 76),
               (0, 214, 255), (170, 66, 244)]
    if show_mask:
        overlay = out.copy()
        for d in detections:
            if d.get("contour") is None:
                continue
            color = palette[d["cls"] % len(palette)]
            overlay[d["mask"] > 0] = color
        out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    for i, d in enumerate(detections):
        if d.get("contour") is None:
            continue
        color = palette[d["cls"] % len(palette)]
        cv2.drawContours(out, [d["contour"]], -1, color, 2)
        x, y, w, h = cv2.boundingRect(d["contour"])
        label = f"{i+1}:{CLASS_NAMES[d['cls']]} {d['conf']:.2f}"
        cv2.putText(out, label, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out


def export_for_ripeness(img_bgr, detections, out_dir=OUTPUT_DIR):
    """
    Flowchart hand-off: Output -> (Ripeness / Defect / Grade).
    For each fruit, save a MASKED crop (background removed, only fruit pixels)
    plus a manifest. This is exactly what the HSV ripeness + defect stage
    consumes. Returns the manifest list.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = os.path.join(out_dir, stamp)
    os.makedirs(session, exist_ok=True)
    manifest = []
    for i, d in enumerate(detections):
        if d.get("contour") is None:
            continue
        # isolate this fruit's pixels only (background -> black)
        masked = cv2.bitwise_and(img_bgr, img_bgr, mask=d["mask"])
        x, y, w, h = cv2.boundingRect(d["contour"])
        crop = masked[y:y + h, x:x + w]
        fname = f"fruit_{i+1:02d}_{CLASS_NAMES[d['cls']]}.png"
        cv2.imwrite(os.path.join(session, fname), crop)
        manifest.append({
            "index": i + 1,
            "fruit_type": CLASS_NAMES[d["cls"]],
            "confidence": round(d["conf"], 4),
            "contour_area_px": round(d["area"], 1),
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "masked_crop": fname,
        })
    with open(os.path.join(session, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return session, manifest


def ripeness_stage(masked_crop_bgr, fruit_type):
    """
    ==================================================================
    -

    ==================================================================
    """
    raise NotImplementedError("Ripeness/defect/grading stage not implemented yet.")


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class PipelineGUI:
    PANEL = 380

    def __init__(self, root):
        self.root = root
        root.title("Fruit Segmentation Pipeline")
        self.model = None
        self.current_img = None          # last loaded BGR image (original)
        self.crops = []                  # per-object crops from preprocessing
        self.crop_rects = []             # (x, y, w, h) of each crop in the original image
        self.crop_page = 0
        self.seg_page = 0
        self.detections = []
        self.show_mask = tk.BooleanVar(value=False)
        self.cap = None                  # active VideoCapture during live mode
        self.live = False                # live-video running flag

        # top bar
        bar = ttk.Frame(root, padding=8)
        bar.pack(fill="x")
        ttk.Button(bar, text="Load Image", command=self.load_image).pack(side="left", padx=4)
        ttk.Button(bar, text="Capture Camera", command=self.capture_camera).pack(side="left", padx=4)
        ttk.Button(bar, text="Run Pipeline", command=self.run_pipeline).pack(side="left", padx=4)
        self.live_btn = ttk.Button(bar, text="Live Video", command=self.start_live)
        self.live_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop_live, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(bar, text="Export for Ripeness", command=self.export).pack(side="left", padx=4)
        ttk.Checkbutton(bar, text="Show mask overlay", variable=self.show_mask,
                        command=self.refresh_segmented).pack(side="left", padx=12)

        # three equal panels: Original | Preprocessed | Segmented
        panels = ttk.Frame(root, padding=8)
        panels.pack(fill="both", expand=False)
        self.panel = {}
        # light-grey placeholder so empty panels aren't big black blocks
        self._placeholder = ImageTk.PhotoImage(
            Image.new("RGB", (self.PANEL, self.PANEL), (238, 238, 238)))
        for col, (key, title) in enumerate([
            ("orig", "1. Original"),
            ("pre", "2. Preprocessed"),
            ("seg", "3. Segmented + Contours"),
        ]):
            frame = ttk.LabelFrame(panels, text=title, padding=4)
            frame.grid(row=0, column=col, padx=6, pady=6, sticky="n")
            # fixed-pixel holder so the label can't balloon when empty
            holder = tk.Frame(frame, width=self.PANEL, height=self.PANEL)
            holder.pack()
            holder.pack_propagate(False)
            lbl = tk.Label(holder, bg="#eeeeee", image=self._placeholder)
            lbl.pack(fill="both", expand=True)
            self.panel[key] = lbl

            if key == "pre":
                nav = ttk.Frame(frame)
                nav.pack(fill="x", pady=(4, 0))
                self.prev_btn = ttk.Button(nav, text="<", width=3, state="disabled",
                                           command=lambda: self.crop_page_step(-1))
                self.prev_btn.pack(side="left")
                self.next_btn = ttk.Button(nav, text=">", width=3, state="disabled",
                                           command=lambda: self.crop_page_step(1))
                self.next_btn.pack(side="right")
                self.crop_label = ttk.Label(nav, text="")
                self.crop_label.pack(side="top")

            if key == "seg":
                nav = ttk.Frame(frame)
                nav.pack(fill="x", pady=(4, 0))
                self.seg_prev_btn = ttk.Button(nav, text="<", width=3, state="disabled",
                                               command=lambda: self.seg_page_step(-1))
                self.seg_prev_btn.pack(side="left")
                self.seg_next_btn = ttk.Button(nav, text=">", width=3, state="disabled",
                                               command=lambda: self.seg_page_step(1))
                self.seg_next_btn.pack(side="right")
                self.seg_label = ttk.Label(nav, text="")
                self.seg_label.pack(side="top")

        # results text
        self.info = tk.Text(root, height=8)
        self.info.pack(fill="x", padx=8, pady=6)
        self.log(f"Model: {MODEL_PATH}")
        self.log("Load an image (or capture from camera), then Run Pipeline.")

    def log(self, msg):
        self.info.insert("end", msg + "\n")
        self.info.see("end")

    def ensure_model(self):
        if self.model is None:
            if not os.path.exists(MODEL_PATH):
                messagebox.showerror("Model not found",
                    f"Could not find the model at:\n{MODEL_PATH}\n"
                    "Edit MODEL_PATH at the top of the script.")
                return False
            self.log("Loading model...")
            self.model = YOLO(MODEL_PATH)
            self.log("Model loaded.")
        return True

    def show(self, key, img_bgr):
        """Render a BGR image into a fixed-size panel, letterboxed to keep aspect."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = self.PANEL / max(h, w)
        rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
        # center on a fixed square canvas (light grey) so all three panels line up
        canvas = np.full((self.PANEL, self.PANEL, 3), 238, dtype=np.uint8)
        yh, xw = rgb.shape[:2]
        y0 = (self.PANEL - yh) // 2
        x0 = (self.PANEL - xw) // 2
        canvas[y0:y0 + yh, x0:x0 + xw] = rgb
        imgtk = ImageTk.PhotoImage(Image.fromarray(canvas))
        lbl = self.panel[key]
        lbl.configure(image=imgtk)
        lbl.image = imgtk

    def clear_panel(self, key):
        self.panel[key].configure(image=self._placeholder)
        self.panel[key].image = self._placeholder

    def show_crops(self):
        if not self.crops:
            self.clear_panel("pre")
            self.crop_label.configure(text="")
            self.prev_btn.configure(state="disabled")
            self.next_btn.configure(state="disabled")
            return
        pages = (len(self.crops) + 3) // 4
        self.crop_page = max(0, min(self.crop_page, pages - 1))
        page = self.crops[self.crop_page * 4:self.crop_page * 4 + 4]
        self.show("pre", build_crop_grid(page, self.PANEL))
        self.crop_label.configure(
            text=f"{len(self.crops)} object(s)  |  page {self.crop_page + 1}/{pages}")
        state = "normal" if pages > 1 else "disabled"
        self.prev_btn.configure(state=state)
        self.next_btn.configure(state=state)

    def crop_page_step(self, step):
        if not self.crops:
            return
        pages = (len(self.crops) + 3) // 4
        self.crop_page = (self.crop_page + step) % pages
        self.show_crops()

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.current_img = cv2.imread(path)
        self.crops = []
        self.crop_rects = []
        self.detections = []
        self.show("orig", self.current_img)
        self.show_crops()
        self.clear_panel("seg")
        self.log(f"Loaded: {os.path.basename(path)}")

    def capture_camera(self):
        """Live preview inside the app (no cv2.imshow window). SPACE captures."""
        cap = cv2.VideoCapture(1)
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
                self.crops = []
                self.crop_rects = []
                self.detections = []
                self.show("orig", self.current_img)
                self.show_crops()
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
        """Full flowchart on one BGR image; updates all three panels.
        Returns (count, per_class). Shared by single-run and live video.
        """
        self.current_img = img_bgr
        
        # 1. Run Preprocessing (returns PreprocessResult with Otsu crop info)
        prep_result = preprocessing.preprocess(img_bgr) 
        
        # Fallback safety check if Otsu found no objects
        if not prep_result.crops:
            self.crops = [prep_result.bgr]
            self.crop_rects = [(0, 0, prep_result.bgr.shape[1], prep_result.bgr.shape[0])]
        else:
            self.crops = prep_result.crops
            self.crop_rects = prep_result.crop_rects

        self.crop_page = 0
        self.seg_page = 0

        self.detections = segment(self.model, self.crops, self.crop_rects, img_bgr.shape)
        
        count = contour_count(self.detections)

        self.show("orig", self.current_img)
        self.show_crops()
        self.refresh_segmented() 

        per_class = {}
        for d in self.detections:
            per_class[CLASS_NAMES[d["cls"]]] = per_class.get(CLASS_NAMES[d["cls"]], 0) + 1
        if log_result:
            self.log(f"\nContour count (fruits detected): {count}")
            if per_class:
                breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(per_class.items()))
                self.log(f"Fruit summary -> {breakdown}")
                self.log("Individual detections:")
                for i, d in enumerate(self.detections):
                    fruit_name = CLASS_NAMES[d["cls"]]
                    confidence = d["conf"]
                    self.log(f"  Fruit {i+1}: {fruit_name} (Confidence: {confidence:.2f})")
                
            else:
                self.log("No fruit detected (image may be a negative/background).")
        return count, per_class

    def run_pipeline(self):
        if self.current_img is None:
            messagebox.showwarning("No image", "Load an image or capture from camera first.")
            return
        if not self.ensure_model():
            return
        self.process_frame(self.current_img)

    def start_live(self):
        if self.live:
            return
        if not self.ensure_model():
            return
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            messagebox.showerror("Camera", "Could not open camera.")
            self.cap = None
            return
        self.live = True
        self.live_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.log("\nLive video started (full pipeline per frame). Click Stop to end.")
        self._live_loop()

    def _live_loop(self):
        if not self.live or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            # full flowchart on this frame; quiet logging to avoid spam
            count, per_class = self.process_frame(frame, log_result=False)
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(per_class.items())) or "none"
            self.panel["seg"].master.master.configure(
                text=f"3. Segmented + Contours  (count: {count} | {summary})")
        # schedule next frame; ~10ms yields to the UI, real rate limited by pipeline speed
        self.root.after(10, self._live_loop)

    def stop_live(self):
        self.live = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.live_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.panel["seg"].master.master.configure(text="3. Segmented + Contours")
        self.log("Live video stopped. Last frame kept - you can Export for Ripeness.")

    def refresh_segmented(self):
        """Redraw the Segmented panel, honouring the mask-overlay toggle.
        Drawn on the ORIGINAL image, since detection ran on it (matching coords).
        When there are no detections, still show the current frame (no contours)
        so live video keeps updating instead of appearing frozen."""
        if self.current_img is None:
            return
        if not self.detections:
            self.show("seg", self.current_img)   # no fruit -> show plain frame
            self.seg_label.configure(text="")
            self.seg_prev_btn.configure(state="disabled")
            self.seg_next_btn.configure(state="disabled")
            return
        pages = len(self.detections) + 1
        self.seg_page = max(0, min(self.seg_page, pages - 1))
        shown = self.detections if self.seg_page == 0 else [self.detections[self.seg_page - 1]]
        annotated = annotate(self.current_img, shown, show_mask=self.show_mask.get())
        if self.seg_page > 0:
            x, y, w, h = cv2.boundingRect(shown[0]["mask"])
            if w and h:
                pad = int(max(w, h) * 0.15)
                H, W = annotated.shape[:2]
                annotated = annotated[max(0, y - pad):min(H, y + h + pad),
                                      max(0, x - pad):min(W, x + w + pad)]
        self.show("seg", annotated)
        self.seg_label.configure(
            text=f"{len(self.detections)} fruit(s)  |  all" if self.seg_page == 0
            else f"fruit {self.seg_page}/{len(self.detections)}")
        self.seg_prev_btn.configure(state="normal")
        self.seg_next_btn.configure(state="normal")

    def seg_page_step(self, step):
        if not self.detections:
            return
        self.seg_page = (self.seg_page + step) % (len(self.detections) + 1)
        self.refresh_segmented()

    def export(self):
        if not self.detections:
            messagebox.showwarning("Nothing to export", "Run the pipeline first.")
            return
        session, manifest = export_for_ripeness(self.current_img, self.detections)
        self.log(f"\nExported {len(manifest)} masked fruit crop(s) for the ripeness stage:")
        self.log(f"  {os.path.abspath(session)}")
        self.log("  (manifest.json lists each fruit's type, area, and crop file.)")

    def on_close(self):
        self.stop_live()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()