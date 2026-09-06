import io
from datetime import datetime

import cv2
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import blemish_detection


PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 16 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=17,
                                spaceAfter=2),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9,
                              textColor=colors.grey, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13,
                             spaceBefore=8, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=10.5,
                             spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9,
                               alignment=TA_LEFT),
        "mono": ParagraphStyle("m", parent=base["Code"], fontName="Courier",
                               fontSize=7.4, leading=8.8),
    }


def _to_flowable(image_bgr, width):
    """numpy BGR (or 2-D mask) -> a reportlab Image scaled to `width`."""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return None

    array = image_bgr
    if array.ndim == 2:
        rgb = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    buffer = io.BytesIO()
    PILImage.fromarray(rgb.astype(np.uint8)).save(buffer, format="PNG")
    buffer.seek(0)

    h, w = rgb.shape[:2]
    return Image(buffer, width=width, height=width * h / w)


def _image_row(pairs, total_width, gap=4 * mm):
    """A row of captioned images that share the available width."""
    pairs = [(img, cap) for img, cap in pairs if img is not None
             and getattr(img, "size", 0)]
    if not pairs:
        return None

    each = (total_width - gap * (len(pairs) - 1)) / len(pairs)
    styles = _styles()

    cells, captions = [], []
    for img, caption in pairs:
        flow = _to_flowable(img, each)
        cells.append(flow if flow is not None else "")
        captions.append(Paragraph(f"<font size=7>{caption}</font>",
                                  styles["body"]))

    table = Table([cells, captions], colWidths=[each] * len(pairs))
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


def _metrics_table(item):
    """The blemish area details, as a two-column key/value block."""
    lo, hi = item.get("area_bounds_px", [0, 0])
    rows = [
        ("Status", str(item.get("status", "-"))),
        ("Defect area", f"{item.get('defect_area_pct', 0):.2f}%"),
        ("Threshold", f"{item.get('defect_threshold_pct', 0):.2f}%"),
        ("Defect pixels", f"{item.get('defect_pixels', 0):,}"),
        ("Fruit pixels", f"{item.get('fruit_pixels', 0):,}"),
        ("Regions", str(item.get("blob_count", 0))),
        ("Largest region", f"{item.get('largest_blob_px', 0):,} px"),
        ("Rejected as shadow", f"{item.get('rejected_px', 0):,} px"),
        ("Cluster levels", str(item.get("cluster_levels", []))),
        ("Area filter", f"{lo:,}..{hi:,} px"),
        ("Healthy chroma", str(item.get("healthy_chroma", 0))),
    ]

    defected = item.get("status") == "Defected"
    table = Table([[k, v] for k, v in rows],
                  colWidths=[42 * mm, CONTENT_WIDTH / 2 - 42 * mm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("TEXTCOLOR", (1, 0), (1, 0),
         colors.HexColor("#b00020") if defected
         else colors.HexColor("#1b7f3b")),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e2e2e2")),
    ]))
    return table


def _regions_table(item):
    """Per-region areas and chroma ratios."""
    blobs = sorted(item.get("blobs") or [], key=lambda b: -b["area_px"])
    if not blobs:
        return None

    fruit_px = item.get("fruit_pixels", 0) or 1
    head = ["#", "Area (px)", "% of fruit", "Chroma ratio"]
    rows = [head]
    for i, blob in enumerate(blobs, start=1):
        rows.append([str(i), f"{blob['area_px']:,}",
                     f"{100.0 * blob['area_px'] / fruit_px:.2f}%",
                     f"{blob['chroma_ratio']:.3f}"])

    width = CONTENT_WIDTH / 2
    table = Table(rows, colWidths=[width * 0.12, width * 0.30,
                                   width * 0.28, width * 0.30],
                  repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
    ]))
    return table


def _summary_table(items):
    head = ["Fruit", "Type", "YOLO said", "Status", "Defect %", "Regions"]
    rows = [head]
    for item in items:
        rows.append([
            str(item.get("number", "-")),
            str(item.get("fruit_type", "-")).title(),
            str(item.get("yolo_fruit_type", "-")).title(),
            item.get("status", "error"),
            ("-" if item.get("defect_area_pct") is None
             else f"{item['defect_area_pct']:.2f}%"),
            str(item.get("blob_count", "-")),
        ])

    table = Table(rows, colWidths=[16 * mm, 30 * mm, 30 * mm, 26 * mm,
                                   24 * mm, 20 * mm], repeatRows=1)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("ALIGN", (3, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i, item in enumerate(items, start=1):
        if item.get("status") == "Defected":
            style.append(("TEXTCOLOR", (3, i), (3, i),
                          colors.HexColor("#b00020")))
        elif item.get("status") == "Good":
            style.append(("TEXTCOLOR", (3, i), (3, i),
                          colors.HexColor("#1b7f3b")))
    table.setStyle(TableStyle(style))
    return table


def build_pdf(items, source_name=None, annotated_image=None):
    if not items:
        raise ValueError("No blemish results to export.")

    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Blemish Detection Report", author="Fruit Ripeness Assessment")

    story = [
        Paragraph("Blemish / Defect Detection Report", styles["title"]),
        Paragraph(
            f"Module 2.4 &middot; k-means darkest-cluster segmentation "
            f"generated {datetime.now():%Y-%m-%d %H:%M}"
            + (f" &middot; source: {source_name}" if source_name else ""),
            styles["sub"]),
    ]

    assessed = [i for i in items if "error" not in i]
    defected = sum(1 for i in assessed if i.get("status") == "Defected")
    story.append(Paragraph(
        f"<b>{len(assessed)}</b> fruit assessed &nbsp;&middot;&nbsp; "
        f"<b>{len(assessed) - defected}</b> good &nbsp;&middot;&nbsp; "
        f"<b>{defected}</b> defected", styles["body"]))
    story.append(Spacer(1, 6))

    if annotated_image is not None and getattr(annotated_image, "size", 0):
        flow = _to_flowable(annotated_image, min(CONTENT_WIDTH, 110 * mm))
        if flow is not None:
            story += [flow, Spacer(1, 4)]

    story += [Paragraph("Summary", styles["h2"]), _summary_table(items),
              PageBreak()]

    for index, item in enumerate(items):
        name = str(item.get("fruit_type", "-")).title()
        heading = f"Fruit {item.get('number', '-')} &ndash; {name}"
        if item.get("fruit_type_overridden"):
            heading += (f" <font size=8 color='#777777'>(corrected from "
                        f"{str(item.get('yolo_fruit_type')).title()})</font>")
        story.append(Paragraph(heading, styles["h2"]))

        if "error" in item:
            story.append(Paragraph(f"<font color='#b00020'>Not assessed: "
                                   f"{item['error']}</font>", styles["body"]))
            if index != len(items) - 1:
                story.append(PageBreak())
            continue

        diagnostics = item.get("diagnostics") or {}
        row = _image_row([
            (diagnostics.get("01_crop"), "Fruit crop"),
            (diagnostics.get("03_darkest_cluster"), "Darkest cluster"),
            (diagnostics.get("04_rejected_as_shadow"), "Rejected as shadow"),
            (diagnostics.get("05_defect_mask"), "Defect mask"),
            (diagnostics.get("06_overlay"), "Overlay"),
        ], CONTENT_WIDTH)
        if row is not None:
            story += [row, Spacer(1, 6)]

        # Blemish area details beside the per-region breakdown.
        regions = _regions_table(item)
        side_by_side = Table(
            [[_metrics_table(item), regions or Paragraph(
                "<font size=8 color='#777777'>No regions passed the "
                "area and chroma filters.</font>", styles["body"])]],
            colWidths=[CONTENT_WIDTH / 2, CONTENT_WIDTH / 2])
        side_by_side.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (0, 0), 0),
            ("RIGHTPADDING", (-1, 0), (-1, 0), 0),
            ("LEFTPADDING", (1, 0), (1, 0), 6),
        ]))
        story += [Paragraph("Blemish area details", styles["h3"]),
                  side_by_side, Spacer(1, 4)]

        story.append(Paragraph("Detection detail", styles["h3"]))
        story.append(Preformatted(
            blemish_detection.format_report(item), styles["mono"]))

        if index != len(items) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()
