import streamlit as st
from typing import Dict, Any

from src.ui.theme import COLORS


def get_recommendation_badge(recommendation: str) -> str:
    """Returns HTML for a styled recommendation badge."""
    badges = {
        "Excellent Buy": f"<span class='val-badge' style='background: rgba(0,255,136,0.15); color: {COLORS['success']}; border: 1px solid {COLORS['success']};'>✨ Excellent Buy</span>",
        "Good Buy": f"<span class='val-badge' style='background: rgba(0,163,255,0.15); color: {COLORS['primary']}; border: 1px solid {COLORS['primary']};'>👍 Good Buy</span>",
        "Fair Purchase": f"<span class='val-badge' style='background: rgba(255,255,255,0.1); color: {COLORS['text_primary']}; border: 1px solid {COLORS['border']};'>⚖️ Fair Purchase</span>",
        "Negotiate": f"<span class='val-badge' style='background: rgba(255,200,87,0.15); color: {COLORS['warning']}; border: 1px solid {COLORS['warning']};'>💬 Negotiate</span>",
        "Avoid": f"<span class='val-badge' style='background: rgba(255,59,48,0.15); color: {COLORS['danger']}; border: 1px solid {COLORS['danger']};'>⚠️ Avoid</span>",
    }
    return badges.get(recommendation, badges["Fair Purchase"])


def get_score_color(score: float) -> str:
    if score >= 85:
        return COLORS["success"]
    if score >= 70:
        return COLORS["primary"]
    if score >= 50:
        return COLORS["warning"]
    return COLORS["danger"]


def render_hero_card(report: Dict[str, Any]):
    """Renders the primary Valuation Hero Card."""
    val = report["valuation_report"]

    est_value = val["estimated_market_value"]
    lower = val["estimated_market_range"]["lower_bound"]
    upper = val["estimated_market_range"]["upper_bound"]

    conf_score = val["confidence"]["score"]
    conf_color = get_score_color(conf_score)

    market_pos = val["market_position"]
    rec = val["recommendation"]

    rec_badge = get_recommendation_badge(rec)

    # Fake a 'Valuation Score' out of 100 based on confidence and recommendation
    val_score = conf_score
    if rec in ["Excellent Buy", "Good Buy"]:
        val_score = min(98, val_score + 5)
    elif rec in ["Avoid", "Negotiate"]:
        val_score = max(20, val_score - 20)

    score_color = get_score_color(val_score)

    st.markdown(
        f"""
    <div class="val-hero-card">
        <div class="val-hero-glow"></div>
        <div style="display: flex; justify-content: space-between; align-items: flex-start; position: relative; z-index: 1;">
            <div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Estimated Market Value</div>
                <div class="val-price">{est_value}</div>
                <div class="val-range">Expected Range: {lower} &mdash; {upper}</div>
            </div>
            <div style="text-align: right;">
                <div style="margin-bottom: 12px;">{rec_badge}</div>
                <div style="background: rgba(0,0,0,0.3); padding: 8px 16px; border-radius: 12px; border: 1px solid {COLORS['border']}; display: inline-block;">
                    <div style="color: {COLORS['text_secondary']}; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 2px;">Valuation Score</div>
                    <div style="color: {score_color}; font-size: 1.5rem; font-weight: 800; font-family: 'Montserrat', sans-serif;">{int(val_score)}<span style="font-size: 0.9rem; color: {COLORS['text_secondary']};">/100</span></div>
                </div>
            </div>
        </div>

        <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid {COLORS['border']}; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; position: relative; z-index: 1;">
            <div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem; margin-bottom: 4px;">Market Position</div>
                <div style="color: {COLORS['text_primary']}; font-weight: 600;">{market_pos}</div>
            </div>
            <div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem; margin-bottom: 4px;">Data Confidence</div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="flex-grow: 1; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
                        <div style="width: {conf_score}%; height: 100%; background: {conf_color}; border-radius: 3px;"></div>
                    </div>
                    <span style="color: {conf_color}; font-weight: 600; font-size: 0.9rem;">{conf_score}%</span>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_spec_card(summary: Dict[str, Any]):
    """Renders the Vehicle Specifications."""
    st.markdown(
        f"""
    <div class="val-spec-grid">
        <div class="val-spec-item">
            <div class="val-spec-label">Engine</div>
            <div class="val-spec-value">{summary.get('engineSize', 'N/A')} L</div>
        </div>
        <div class="val-spec-item">
            <div class="val-spec-label">Transmission</div>
            <div class="val-spec-value">{summary.get('transmission', 'N/A')}</div>
        </div>
        <div class="val-spec-item">
            <div class="val-spec-label">Fuel Type</div>
            <div class="val-spec-value">{summary.get('fuelType', 'N/A')}</div>
        </div>
        <div class="val-spec-item">
            <div class="val-spec-label">Odometer</div>
            <div class="val-spec-value">{summary.get('mileage', 0):,} km</div>
        </div>
        <div class="val-spec-item">
            <div class="val-spec-label">Model Year</div>
            <div class="val-spec-value">{summary.get('year', 'N/A')}</div>
        </div>
        <div class="val-spec-item">
            <div class="val-spec-label">Vehicle Age</div>
            <div class="val-spec-value">{summary.get('car_age', 'N/A')} Years</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_similar_vehicles(similar_vehicles: list):
    """Renders similar vehicles as responsive cards."""
    if not similar_vehicles:
        st.info("No similar vehicles found in the current market database.")
        return

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Show up to 4 similar vehicles
    display_items = similar_vehicles[:4]
    cols = st.columns(len(display_items))

    for i, row in enumerate(display_items):
        with cols[i]:
            price = row.get("price", "N/A")
            st.markdown(
                f"""
            <div class="val-similar-card">
                <div style="font-size: 0.8rem; color: {COLORS['primary']}; font-weight: 600; margin-bottom: 4px;">{row.get('year', '')}</div>
                <div style="color: {COLORS['text_primary']}; font-weight: 600; font-size: 1.05rem; line-height: 1.2; margin-bottom: 4px;">
                    {row.get('brand', '')} {row.get('model', '')}
                </div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem; margin-bottom: 12px;">
                    {row.get('transmission', '')} &middot; {row.get('fuelType', '')} &middot; {row.get('mileage', 0):,} km
                </div>
                <div style="font-family: 'Montserrat', sans-serif; font-weight: 700; color: {COLORS['text_primary']}; font-size: 1.1rem;">
                    {price}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
