import streamlit as st
from typing import Dict, Any

from src.ui.theme import COLORS


def render_ai_summary(report: Dict[str, Any]):
    """Renders the AI Valuation Summary prominently."""
    summary_text = report["valuation_report"]["ai_summary"]

    st.markdown(
        f"""
    <div class="val-summary-card">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{COLORS['primary']}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
            <span style="color: {COLORS['primary']}; font-weight: 700; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px;">AI Valuation Summary</span>
        </div>
        <div class="val-summary-text">
            {summary_text}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_factors_cards(report: Dict[str, Any]):
    """Renders Positive and Negative Contributors."""
    explanation = report["valuation_report"]["explanation"]
    pos_factors = explanation.get("major_positive_factors", [])
    neg_factors = explanation.get("major_negative_factors", [])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
        <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-radius: 12px; padding: 20px; height: 100%;">
            <div style="color: {COLORS['success']}; font-weight: 700; font-size: 1.1rem; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                Positive Contributors
            </div>
            <ul class="val-factor-list">
        """,
            unsafe_allow_html=True,
        )

        if pos_factors:
            for factor in pos_factors:
                st.markdown(
                    f"""
                <li class="val-factor-item">
                    <div style="color: {COLORS['success']}; margin-top: 2px;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                    </div>
                    <div>
                        <div class="val-factor-title">{factor['feature']}</div>
                        <div class="val-factor-desc">{factor['explanation']}</div>
                    </div>
                </li>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<div style='color: {COLORS['text_secondary']}; font-size: 0.9rem;'>No major positive factors identified.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</ul></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            f"""
        <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-radius: 12px; padding: 20px; height: 100%;">
            <div style="color: {COLORS['danger']}; font-weight: 700; font-size: 1.1rem; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                Negative Contributors
            </div>
            <ul class="val-factor-list">
        """,
            unsafe_allow_html=True,
        )

        if neg_factors:
            for factor in neg_factors:
                st.markdown(
                    f"""
                <li class="val-factor-item">
                    <div style="color: {COLORS['danger']}; margin-top: 2px;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                    </div>
                    <div>
                        <div class="val-factor-title">{factor['feature']}</div>
                        <div class="val-factor-desc">{factor['explanation']}</div>
                    </div>
                </li>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<div style='color: {COLORS['text_secondary']}; font-size: 0.9rem;'>No major negative factors identified.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</ul></div>", unsafe_allow_html=True)


def render_risk_assessment(report: Dict[str, Any]):
    """Renders Risk Assessment Indicators."""
    risks = report["valuation_report"]["risk_assessment"]

    st.markdown(
        """
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
    """,
        unsafe_allow_html=True,
    )

    def _get_risk_color(level):
        if level == "Low":
            return COLORS["success"]
        if level == "Medium":
            return COLORS["warning"]
        return COLORS["danger"]

    for risk_type, level in risks.items():
        color = _get_risk_color(level)
        st.markdown(
            f"""
        <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-left: 3px solid {color}; border-radius: 8px; padding: 12px 16px;">
            <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem; margin-bottom: 4px;">{risk_type}</div>
            <div style="color: {color}; font-weight: 700; font-size: 1rem;">{level}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_buyer_insights(result: Dict[str, Any]):
    """Renders dynamic buyer insights."""
    insights = result.get("decision_report", {}).get("insights", [])
    if not insights:
        return

    st.markdown(
        f"<div style='color: {COLORS['text_primary']}; font-weight: 700; margin-bottom: 12px;'>Buyer Insights</div>",
        unsafe_allow_html=True,
    )

    html = '<div style="display: flex; flex-direction: column; gap: 10px;">'
    for insight in insights:
        html += f"""<div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 12px 16px; display: flex; align-items: flex-start; gap: 12px;">
            <div style="color: {COLORS['primary']}; margin-top: 2px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
            </div>
            <div>
                <div style="color: {COLORS['text_primary']}; font-weight: 600; font-size: 0.95rem;">{insight["insight"]}</div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.85rem; margin-top: 4px;">{insight["explanation"]}</div>
            </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_negotiation_assistant(result: Dict[str, Any]):
    """Renders negotiation assistant if asking price provided."""
    nav = result.get("decision_report", {}).get("negotiation_assistant", {})
    if not nav or not nav.get("is_available"):
        return

    st.markdown(
        f"<div style='color: {COLORS['text_primary']}; font-weight: 700; margin-bottom: 12px;'>Negotiation Assistant</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-radius: 12px; padding: 16px;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
            <div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem;">Seller Asking Price</div>
                <div style="color: {COLORS['danger']}; font-weight: 700; font-size: 1.1rem;">{nav['seller_asking_price']}</div>
            </div>
            <div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem;">Estimated Fair Value</div>
                <div style="color: {COLORS['success']}; font-weight: 700; font-size: 1.1rem;">{nav['estimated_fair_value']}</div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; padding-top: 16px; border-top: 1px dashed {COLORS['border']};">
            <div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem;">Suggested Offer</div>
                <div style="color: {COLORS['primary']}; font-weight: 700; font-size: 1.1rem;">{nav['suggested_initial_offer']}</div>
            </div>
            <div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem;">Max Recommended</div>
                <div style="color: {COLORS['text_primary']}; font-weight: 700; font-size: 1.1rem;">{nav['maximum_recommended_offer']}</div>
            </div>
        </div>
        <div style="background: rgba(0,163,255,0.05); border-radius: 8px; padding: 12px;">
            <div style="color: {COLORS['text_primary']}; font-size: 0.9rem; font-weight: 600;">Difficulty: {nav['negotiation_difficulty']}</div>
            <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem; margin-top: 4px;">{nav['reasoning']}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_ownership_costs(result: Dict[str, Any]):
    """Renders ownership costs."""
    costs = result.get("decision_report", {})
    if "total_5y" not in costs:
        return

    st.markdown(
        f"""
    <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['border']}; border-radius: 12px; padding: 16px;">
        <div style="color: {COLORS['text_secondary']}; font-size: 0.85rem; margin-bottom: 16px;">
            Estimated planning costs for the next 5 years based on ~{costs['assumptions']['annual_distance_km']} km/year.
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: {COLORS['text_primary']};">Insurance</span>
            <span style="color: {COLORS['text_secondary']};">{costs['insurance_5y']}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: {COLORS['text_primary']};">Maintenance</span>
            <span style="color: {COLORS['text_secondary']};">{costs['maintenance_5y']}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: {COLORS['text_primary']};">Fuel</span>
            <span style="color: {COLORS['text_secondary']};">{costs['fuel_5y']}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 16px;">
            <span style="color: {COLORS['text_primary']};">Registration & Misc</span>
            <span style="color: {COLORS['text_secondary']};">{costs['registration_misc_5y']}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding-top: 12px; border-top: 1px solid {COLORS['border']};">
            <span style="color: {COLORS['text_primary']}; font-weight: 700;">Total 5-Year Cost</span>
            <span style="color: {COLORS['primary']}; font-weight: 700;">{costs['total_5y']}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
