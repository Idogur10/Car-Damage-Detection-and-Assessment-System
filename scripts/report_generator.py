"""
PDF Report Generator for Car Damage Assessment

Generates a professional PDF report combining:
  1. YOLO detections (what damages were found)
  2. CLIP analysis (severity + location of each damage)
  3. Cost estimates (repair cost range per damage + total)
  4. Annotated image (original image with bounding boxes)
  5. Cropped damage images (close-up of each damage)

Uses fpdf2 — a lightweight PDF library that doesn't need external dependencies.


=== OUTPUT FORMAT ===

Page 1: Summary
  - Input image with all bounding boxes drawn
  - Table: all damages with type, severity, location, cost
  - Total estimated repair cost

Page 2+: One section per damage (if many damages)
  - Cropped image of the damage
  - Detailed info: type, severity, confidence, location, cost, repair method
"""

from fpdf import FPDF
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os


class DamageReportPDF(FPDF):
    """
    Custom PDF class with header and footer.

    Inherits from FPDF and overrides header() and footer() methods.
    These are called automatically by fpdf2 on every page.
    """

    def header(self):
        """Called automatically at the top of every page."""
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Car Damage Assessment Report", align="C", new_x="LMARGIN", new_y="NEXT")
        # Draw a line under the header
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)  # Add some space after the line

    def footer(self):
        """Called automatically at the bottom of every page."""
        self.set_y(-15)  # Position 15mm from bottom
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_report(
    image_path,
    detections,
    cost_summary,
    output_path=None,
):
    """
    Generate a PDF report for one car image.

    Args:
        image_path: string or Path — path to the original car image
        detections: list of dicts, each with:
            - damage_type: "scratch", "dent", etc.
            - severity: "minor", "moderate", "severe"
            - severity_confidence: float 0-1
            - location_label: "front door", "hood", etc.
            - location_confidence: float 0-1
            - bbox: dict with x1, y1, x2, y2
            - crop: PIL Image of the cropped damage region
            - cost: dict from estimate_cost() with min_cost, max_cost, method
        cost_summary: dict from estimate_total_cost() with total_min, total_max, total_average
        output_path: where to save the PDF. If None, auto-generates a name.

    Returns:
        Path to the generated PDF file.
    """

    image_path = Path(image_path)

    # Auto-generate output path if not provided
    if output_path is None:
        output_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"damage_report_{image_path.stem}_{timestamp}.pdf"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create PDF
    pdf = DamageReportPDF()
    pdf.alias_nb_pages()  # Enable {nb} placeholder for total pages in footer
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---------------------------------------------------------------
    # PAGE 1: Summary
    # ---------------------------------------------------------------
    pdf.add_page()

    # --- Report metadata ---
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Image: {image_path.name}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Damages found: {len(detections)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # --- Annotated image ---
    # Draw bounding boxes on the original image and embed it in the PDF
    annotated_image = _draw_annotations(image_path, detections)

    # Save annotated image to a temp file (fpdf2 needs a file path, not PIL image)
    temp_dir = tempfile.mkdtemp()
    annotated_path = os.path.join(temp_dir, "annotated.jpg")
    annotated_image.save(annotated_path, quality=90)

    # Calculate image dimensions to fit in the PDF
    # PDF page width is 190mm (210mm page - 10mm margins on each side)
    max_width = 190
    img_width, img_height = annotated_image.size
    aspect_ratio = img_height / img_width
    display_width = min(max_width, 160)  # cap at 160mm for readability
    display_height = display_width * aspect_ratio

    # Cap height too — don't let a tall image push everything off the page
    if display_height > 100:
        display_height = 100
        display_width = display_height / aspect_ratio

    # Center the image
    x_offset = (210 - display_width) / 2
    pdf.image(annotated_path, x=x_offset, w=display_width, h=display_height)
    pdf.ln(5)

    # --- Damage Summary Table ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Damage Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    col_widths = [8, 28, 22, 35, 45, 52]  # #, Type, Severity, Location, Cost Range, Method
    headers = ["#", "Type", "Severity", "Location", "Cost Range", "Repair Method"]

    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 7, header, border=1, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 9)
    for idx, det in enumerate(detections):
        row_num = str(idx + 1)
        damage_type = det["damage_type"].replace("_", " ")
        severity = det["severity"]
        location = det["location_label"]

        if det.get("cost"):
            cost_range = f"${det['cost']['min_cost']:,}-${det['cost']['max_cost']:,}"
            method = det["cost"]["method"]
        else:
            cost_range = "N/A"
            method = "N/A"

        row_data = [row_num, damage_type, severity, location, cost_range, method]

        for i, cell_text in enumerate(row_data):
            pdf.cell(col_widths[i], 6, cell_text, border=1, align="C")
        pdf.ln()

    pdf.ln(5)

    # --- Total Cost ---
    pdf.set_font("Helvetica", "B", 11)
    if cost_summary["count"] > 0:
        pdf.cell(
            0, 8,
            f"Estimated Total: ${cost_summary['total_min']:,} - ${cost_summary['total_max']:,}",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(
            0, 6,
            f"Best estimate (midpoint): ${cost_summary['total_average']:,.0f}",
            new_x="LMARGIN", new_y="NEXT",
        )
    else:
        pdf.cell(0, 8, "No cost estimates available", new_x="LMARGIN", new_y="NEXT")

    # ---------------------------------------------------------------
    # PAGE 2+: Individual Damage Details
    # ---------------------------------------------------------------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Individual Damage Details", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Keep track of temp files for cleanup
    temp_crop_paths = []

    for idx, det in enumerate(detections):
        # Check if we need a new page (leave room for image + text)
        if pdf.get_y() > 220:
            pdf.add_page()

        # --- Damage header ---
        pdf.set_font("Helvetica", "B", 10)
        damage_label = det["damage_type"].replace("_", " ").title()
        pdf.cell(0, 7, f"Damage #{idx + 1}: {damage_label}", new_x="LMARGIN", new_y="NEXT")

        # --- Crop image ---
        if det.get("crop") is not None:
            crop_path = os.path.join(temp_dir, f"crop_{idx}.jpg")
            det["crop"].save(crop_path, quality=85)
            temp_crop_paths.append(crop_path)

            # Display crop at reasonable size
            crop_w, crop_h = det["crop"].size
            crop_aspect = crop_h / crop_w
            display_w = 50  # 50mm wide
            display_h = display_w * crop_aspect
            if display_h > 40:
                display_h = 40
                display_w = display_h / crop_aspect

            pdf.image(crop_path, w=display_w, h=display_h)
            pdf.ln(2)

        # --- Details ---
        pdf.set_font("Helvetica", "", 9)

        details = [
            f"Type: {det['damage_type'].replace('_', ' ')}",
            f"Severity: {det['severity']} (confidence: {det['severity_confidence']:.0%})",
            f"Location: {det['location_label']} (confidence: {det['location_confidence']:.0%})",
        ]

        if det.get("cost"):
            details.append(f"Cost estimate: ${det['cost']['min_cost']:,} - ${det['cost']['max_cost']:,}")
            details.append(f"Repair method: {det['cost']['method']}")

        for line in details:
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")

        pdf.ln(5)

    # --- Disclaimer ---
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0, 4,
        "Disclaimer: Cost estimates are approximate and based on US market averages. "
        "Actual repair costs depend on vehicle make/model, location, parts availability, "
        "and shop rates. Severity and location are assessed by CLIP zero-shot classification "
        "and may not be fully accurate. This report is for informational purposes only.",
    )

    # --- Save PDF ---
    pdf.output(str(output_path))

    # --- Cleanup temp files ---
    try:
        os.remove(annotated_path)
        for crop_path in temp_crop_paths:
            os.remove(crop_path)
        os.rmdir(temp_dir)
    except OSError:
        pass  # temp files will be cleaned up by OS eventually

    print(f"PDF report saved: {output_path}")
    return output_path


def _draw_annotations(image_path, detections):
    """
    Draw bounding boxes and labels on the car image.

    Args:
        image_path: Path to the original image
        detections: list of detection dicts with bbox, damage_type, severity

    Returns:
        PIL Image with annotations drawn on it
    """

    image = Image.open(str(image_path)).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Color map for severity levels
    severity_colors = {
        "minor": (0, 200, 0),       # green
        "moderate": (255, 165, 0),   # orange
        "severe": (255, 0, 0),       # red
    }

    for idx, det in enumerate(detections):
        bbox = det["bbox"]
        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        # Pick color based on severity
        color = severity_colors.get(det["severity"], (255, 255, 0))

        # Draw bounding box (3 pixels thick)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # Draw label above the box
        label = f"{det['damage_type']} - {det['severity']}"
        # Draw text background for readability
        text_bbox = draw.textbbox((x1, y1 - 15), label)
        draw.rectangle(text_bbox, fill=(0, 0, 0))
        draw.text((x1, y1 - 15), label, fill=color)

    return image


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Report generator module loaded.")
    print("Use generate_report() from the full pipeline.")
    print("Or run full_pipeline.py for end-to-end processing.")
