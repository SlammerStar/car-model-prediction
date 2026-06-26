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
            "AI-Powered Valuation & Decision Report",
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

    val = copy.deepcopy(report.get("valuation_report", {}))
    summary = report.get("input_summary", {})
    decision = copy.deepcopy(report.get("decision_report", {}))

    # fpdf2's default helvetica doesn't support the Rupee symbol
    def sanitize(text):
        if isinstance(text, str):
            return text.replace("₹", "Rs. ")
        return text

    val["estimated_market_value"] = sanitize(val.get("estimated_market_value", ""))
    if "estimated_market_range" in val:
        val["estimated_market_range"]["lower_bound"] = sanitize(
            val["estimated_market_range"].get("lower_bound", "")
        )
        val["estimated_market_range"]["upper_bound"] = sanitize(
            val["estimated_market_range"].get("upper_bound", "")
        )
    val["ai_summary"] = sanitize(val.get("ai_summary", ""))

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
    pdf.cell(60, 8, "Estimated Market Value", border="TLR", fill=True)
    pdf.cell(60, 8, "Deal Score", border="TLR", fill=True)
    pdf.cell(
        60, 8, "Recommendation", border="TLR", fill=True, new_x="LMARGIN", new_y="NEXT"
    )

    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(60, 12, val.get("estimated_market_value", ""), border="LR")
    deal_score_str = f"{decision.get('deal_score', 'N/A')}/100"
    pdf.cell(60, 12, deal_score_str, border="LR", align="C")
    final_rec = decision.get("final_recommendation", val.get("recommendation", "N/A"))
    pdf.cell(60, 12, final_rec, border="LR", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    range_str = "N/A"
    if "estimated_market_range" in val:
        range_str = f"Range: {val['estimated_market_range'].get('lower_bound', '')} - {val['estimated_market_range'].get('upper_bound', '')}"
    pdf.cell(60, 8, range_str, border="BLR")
    conf_str = f"Confidence: {val.get('confidence', {}).get('score', 0)}% ({val.get('confidence', {}).get('label', '')})"
    pdf.cell(60, 8, conf_str, border="BLR", align="C")
    pdf.cell(60, 8, "", border="BLR", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # AI Summary
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "AI Valuation Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, val.get("ai_summary", ""), new_x="LMARGIN", new_y="NEXT")
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

    for r_type, r_level in val.get("risk_assessment", {}).items():
        pdf.set_x(105)
        pdf.cell(50, 6, f"{r_type}:")
        pdf.cell(40, 6, str(r_level), new_x="LMARGIN", new_y="NEXT")

    current_y = pdf.get_y()
    specs_end_y = y_start + 24
    if current_y < specs_end_y:
        pdf.set_y(specs_end_y)

    pdf.ln(8)

    # Buyer Insights & Ownership Costs
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(90, 8, "Buyer Insights", new_x="RIGHT")
    pdf.cell(90, 8, "Estimated Ownership Costs", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    y_start = pdf.get_y()

    buyer_insights = decision.get("buyer_insights", [])
    if buyer_insights:
        for idx, insight in enumerate(buyer_insights[:3]):
            pdf.multi_cell(
                85, 6, f"- {insight.get('insight', '')}", new_x="LMARGIN", new_y="NEXT"
            )
    else:
        pdf.cell(85, 6, "N/A", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(y_start)
    pdf.set_x(105)

    ownership = decision.get("ownership_forecast", {})
    if ownership:
        pdf.cell(50, 6, "Annual Fuel Cost:")
        pdf.cell(
            40,
            6,
            sanitize(ownership.get("annual_fuel_cost", "N/A")),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_x(105)
        pdf.cell(50, 6, "Annual Maintenance:")
        pdf.cell(
            40,
            6,
            sanitize(ownership.get("annual_maintenance_cost", "N/A")),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_x(105)
        pdf.cell(50, 6, "Total 5-Year Cost:")
        pdf.cell(
            40,
            6,
            sanitize(ownership.get("total_5_year_cost", "N/A")),
            new_x="LMARGIN",
            new_y="NEXT",
        )
    else:
        pdf.cell(40, 6, "N/A", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)

    # Factors
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Valuation Factors", new_x="LMARGIN", new_y="NEXT")

    pos_factors = val.get("explanation", {}).get("major_positive_factors", [])
    if pos_factors:
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 6, "Positive Contributors", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        for f in pos_factors:
            pdf.multi_cell(
                0,
                6,
                f"- {f['feature']}: {f['explanation']}",
                new_x="LMARGIN",
                new_y="NEXT",
            )

    pdf.ln(4)

    neg_factors = val.get("explanation", {}).get("major_negative_factors", [])
    if neg_factors:
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 6, "Negative Contributors", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        for f in neg_factors:
            pdf.multi_cell(
                0,
                6,
                f"- {f['feature']}: {f['explanation']}",
                new_x="LMARGIN",
                new_y="NEXT",
            )

    pdf.ln(8)

    # Future Value Forecast & Similar Vehicles
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Future Value Forecast", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    forecast = decision.get("value_forecast", [])
    if forecast:
        for f_year in forecast[:3]:
            pdf.cell(
                60,
                6,
                f"Year {f_year['year']} ({f_year['retention_percentage']}%):",
                border=0,
            )
            pdf.cell(
                60,
                6,
                sanitize(f_year["estimated_value"]),
                border=0,
                new_x="LMARGIN",
                new_y="NEXT",
            )
    else:
        pdf.cell(0, 6, "Forecast not available.", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)

    similar = decision.get("similar_vehicles", [])
    if similar:
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, "Similar Market Alternatives", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        for sv in similar[:3]:
            pdf.cell(
                0,
                6,
                f"- {sv['label']} - {sanitize(sv['estimated_value'])} (Match: {sv['match_score']}%)",
                new_x="LMARGIN",
                new_y="NEXT",
            )

    pdf.ln(8)

    # System Metadata
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, "System Metadata", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 8)
    pdf.cell(
        0,
        5,
        f"Model Version: v1.2.0 | Dataset: DRIVEIQ 2026 | Confidence: {val.get('confidence', {}).get('score', 0)}%",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    # Save to temp file
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    pdf.output(temp_path)

    return temp_path
