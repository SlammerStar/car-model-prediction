import os
import tempfile
from datetime import datetime
from typing import Dict, Any
from fpdf import FPDF


class ValuationPDF(FPDF):
    def header(self):
        # Logo / Brand
        self.set_font("helvetica", "B", 18)
        self.set_text_color(0, 163, 255)  # Primary Blue
        self.cell(0, 10, "DRIVEIQ", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("helvetica", "I", 10)
        self.set_text_color(100, 100, 100)
        self.cell(
            0,
            6,
            "AI-Powered Valuation Report",
            new_x="LMARGIN",
            new_y="NEXT",
            align="L",
        )
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        page_str = f"Page {self.page_no()} / {{nb}}"
        self.cell(0, 10, page_str, align="C")


def generate_valuation_pdf(report: Dict[str, Any]) -> str:
    """Generates a PDF valuation report and returns the path to the temporary file."""
    import copy

    val = copy.deepcopy(report["valuation_report"])
    summary = report["input_summary"]

    # fpdf2's default helvetica doesn't support the Rupee symbol
    def sanitize(text):
        if isinstance(text, str):
            return text.replace("₹", "Rs. ")
        return text

    val["estimated_market_value"] = sanitize(val["estimated_market_value"])
    val["estimated_market_range"]["lower_bound"] = sanitize(
        val["estimated_market_range"]["lower_bound"]
    )
    val["estimated_market_range"]["upper_bound"] = sanitize(
        val["estimated_market_range"]["upper_bound"]
    )
    val["ai_summary"] = sanitize(val["ai_summary"])

    pdf = ValuationPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Section
    title = f"{summary.get('year', '')} {summary.get('brand', '')} {summary.get('model', '')}"
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 6, f"Generated on: {now_str}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Hero Section
    pdf.set_fill_color(245, 245, 245)
    pdf.set_draw_color(200, 200, 200)

    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(0, 163, 255)
    pdf.cell(90, 8, "Estimated Market Value", border="TLR", fill=True)
    pdf.cell(
        90, 8, "Recommendation", border="TLR", fill=True, new_x="LMARGIN", new_y="NEXT"
    )

    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(90, 12, val["estimated_market_value"], border="LR")
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(90, 12, val["recommendation"], border="LR", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        90,
        8,
        f"Range: {val['estimated_market_range']['lower_bound']} - {val['estimated_market_range']['upper_bound']}",
        border="BLR",
    )
    pdf.cell(
        90,
        8,
        f"Confidence: {val['confidence']['score']}% ({val['confidence']['label']})",
        border="BLR",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(8)

    # AI Summary
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "AI Valuation Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, val["ai_summary"])
    pdf.ln(8)

    # Vehicle Specs & Risk
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(90, 8, "Vehicle Specifications", new_x="RIGHT")
    pdf.cell(90, 8, "Risk Assessment", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    y_start = pdf.get_y()

    # Specs
    pdf.cell(40, 6, "Odometer:")
    pdf.cell(50, 6, f"{summary.get('mileage', 0):,} km", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(40, 6, "Transmission:")
    pdf.cell(50, 6, summary.get("transmission", "N/A"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(40, 6, "Fuel Type:")
    pdf.cell(50, 6, summary.get("fuelType", "N/A"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(40, 6, "Engine:")
    pdf.cell(
        50, 6, f"{summary.get('engineSize', 'N/A')} L", new_x="LMARGIN", new_y="NEXT"
    )

    # Reset Y for Risks
    pdf.set_y(y_start)

    for r_type, r_level in val["risk_assessment"].items():
        pdf.set_x(105)  # LMARGIN is 10 + 90 = 100 + 5 padding
        pdf.cell(50, 6, f"{r_type}:")
        pdf.cell(40, 6, str(r_level), new_x="LMARGIN", new_y="NEXT")

    # Ensure we drop below both columns before continuing
    current_y = pdf.get_y()
    specs_end_y = y_start + 24  # 4 lines of specs
    if current_y < specs_end_y:
        pdf.set_y(specs_end_y)

    pdf.ln(10)

    # Factors
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 8, "Valuation Factors", new_x="LMARGIN", new_y="NEXT")

    pos_factors = val["explanation"].get("major_positive_factors", [])
    if pos_factors:
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 6, "Positive Contributors", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        for f in pos_factors:
            pdf.multi_cell(0, 6, f"• {f['feature']}: {f['explanation']}")

    pdf.ln(4)

    neg_factors = val["explanation"].get("major_negative_factors", [])
    if neg_factors:
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 6, "Negative Contributors", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        for f in neg_factors:
            pdf.multi_cell(0, 6, f"• {f['feature']}: {f['explanation']}")

    # Save to temp file
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    pdf.output(temp_path)

    return temp_path
