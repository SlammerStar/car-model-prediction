import streamlit as st
from typing import Dict, Any

from src.ui.theme import load_dashboard_css, COLORS
from src.ui.valuation_cards import (
    render_hero_card,
    render_spec_card,
    render_similar_vehicles,
)
from src.ui.charts import (
    render_price_gauge,
    render_future_value_timeline,
    render_feature_contributions,
)
from src.ui.report_components import (
    render_ai_summary,
    render_factors_cards,
    render_risk_assessment,
    render_buyer_insights,
    render_negotiation_assistant,
    render_ownership_costs,
)
from src.ui.export import generate_valuation_pdf


def render_valuation_dashboard(result: Dict[str, Any], similar_vehicles: list = None):
    """
    Main orchestrator for the Premium AI Valuation Dashboard.
    Replaces the inline app.py rendering logic.
    """
    load_dashboard_css()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 1. Hero Section (Full width)
    render_hero_card(result)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 2. Main Tabs
    tab_overview, tab_market, tab_details = st.tabs(
        ["💡 AI Valuation Overview", "📈 Market Intelligence", "🔍 Vehicle Details"]
    )

    with tab_overview:
        col1, col2 = st.columns([1.5, 1])

        with col1:
            render_ai_summary(result)
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            render_buyer_insights(result)
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            render_factors_cards(result)

        with col2:
            st.markdown(
                f"<div style='color: {COLORS['text_primary']}; font-weight: 700; margin-bottom: 12px;'>Price Position</div>",
                unsafe_allow_html=True,
            )
            render_price_gauge(result)

            st.markdown(
                f"<div style='color: {COLORS['text_primary']}; font-weight: 700; margin-bottom: 12px; margin-top: 20px;'>Risk Assessment</div>",
                unsafe_allow_html=True,
            )
            render_risk_assessment(result)

            st.markdown(
                f"<div style='color: {COLORS['text_primary']}; font-weight: 700; margin-bottom: 12px; margin-top: 20px;'>Vehicle Specifications</div>",
                unsafe_allow_html=True,
            )
            render_spec_card(result["input_summary"])

            st.markdown(
                f"<div style='color: {COLORS['text_primary']}; font-weight: 700; margin-bottom: 12px; margin-top: 20px;'>Estimated 5-Year Ownership</div>",
                unsafe_allow_html=True,
            )
            render_ownership_costs(result)

    with tab_market:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                f"<div style='color: {COLORS['text_primary']}; font-weight: 700; margin-bottom: 12px;'>Feature Impact Analysis</div>",
                unsafe_allow_html=True,
            )
            render_feature_contributions(result)

        with col2:
            st.markdown(
                f"<div style='color: {COLORS['text_primary']}; font-weight: 700; margin-bottom: 12px;'>Future Value Projection</div>",
                unsafe_allow_html=True,
            )
            render_future_value_timeline(result)

            if (
                result.get("decision_report", {})
                .get("negotiation_assistant", {})
                .get("is_available", False)
            ):
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                render_negotiation_assistant(result)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='color: {COLORS['text_primary']}; font-weight: 700; margin-bottom: 12px;'>Similar Vehicles in Market</div>",
            unsafe_allow_html=True,
        )
        render_similar_vehicles(similar_vehicles)

    with tab_details:
        st.markdown("### Raw Valuation Data")
        st.json(result["valuation_report"])

    # 3. Export Action
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.divider()

    col_export, col_empty = st.columns([1, 3])
    with col_export:
        if st.button(
            "Download Professional PDF Report", type="primary", use_container_width=True
        ):
            with st.spinner("Generating PDF..."):
                pdf_path = generate_valuation_pdf(result)
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="Click to Save PDF",
                    data=pdf_bytes,
                    file_name=f"DRIVEIQ_Valuation_{result['input_summary']['brand']}_{result['input_summary']['model']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
