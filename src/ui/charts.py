import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

from src.ui.theme import COLORS


def render_price_gauge(report: Dict[str, Any]):
    """Renders a clean horizontal price range visualization."""
    val = report["valuation_report"]
    pred_raw = val["estimated_market_value_raw"]
    lower_raw = val["estimated_market_range"]["lower_raw"]
    upper_raw = val["estimated_market_range"]["upper_raw"]

    # Calculate percentages for the gauge
    total_range = upper_raw - lower_raw
    if total_range == 0:
        total_range = 1

    pred_pct = ((pred_raw - lower_raw) / total_range) * 100

    st.markdown(
        f"""
    <div style="margin-top: 10px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem; color: {COLORS['text_secondary']};">
            <div>Lower Bound<br><span style="color: {COLORS['text_primary']}; font-weight: 600;">{val['estimated_market_range']['lower_bound']}</span></div>
            <div style="text-align: right;">Upper Bound<br><span style="color: {COLORS['text_primary']}; font-weight: 600;">{val['estimated_market_range']['upper_bound']}</span></div>
        </div>
        <div style="position: relative; height: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; height: 100%; width: 100%; background: linear-gradient(90deg, {COLORS['danger']} 0%, {COLORS['warning']} 20%, {COLORS['success']} 50%, {COLORS['warning']} 80%, {COLORS['danger']} 100%); opacity: 0.6;"></div>
        </div>
        <div style="position: relative; margin-top: -14px;">
            <div style="position: absolute; left: {pred_pct}%; transform: translateX(-50%); text-align: center;">
                <div style="width: 2px; height: 20px; background: {COLORS['text_primary']}; margin: 0 auto; box-shadow: 0 0 4px rgba(0,0,0,0.5);"></div>
                <div style="background: {COLORS['card_bg']}; border: 1px solid {COLORS['primary']}; color: {COLORS['text_primary']}; font-size: 0.8rem; font-weight: 700; padding: 2px 8px; border-radius: 12px; margin-top: 4px; white-space: nowrap; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                    Predicted
                </div>
            </div>
        </div>
    </div>
    <div style="height: 30px;"></div>
    """,
        unsafe_allow_html=True,
    )


def render_future_value_timeline(report: Dict[str, Any]):
    """Renders estimated future resale values."""
    val = report["valuation_report"]
    current_price = val["estimated_market_value_raw"]

    # We simulate future value using a standard depreciation curve (approx 10-15% per year)
    # The actual stats could provide this, but we'll use a heuristic for the visualization.
    years = [0, 1, 2, 3, 4, 5]
    labels = ["Current", "1 Year", "2 Years", "3 Years", "4 Years", "5 Years"]

    # Assume 12% annual depreciation for this example visualization
    depreciation_rate = 0.12
    values = [current_price * ((1 - depreciation_rate) ** y) for y in years]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=3, shape="spline", smoothing=1.3),
            marker=dict(
                size=8,
                color=COLORS["card_bg"],
                line=dict(color=COLORS["primary"], width=2),
            ),
            fill="tozeroy",
            fillcolor="rgba(0,163,255,0.1)",
            hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color=COLORS["text_secondary"]),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["border"],
            color=COLORS["text_secondary"],
            tickformat="₹,.0f",
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_feature_contributions(report: Dict[str, Any]):
    """Renders a clean horizontal bar chart for feature impacts."""
    explanation = report["valuation_report"]["explanation"]
    pos = explanation.get("major_positive_factors", [])
    neg = explanation.get("major_negative_factors", [])

    # Combine and sort by absolute contribution
    all_factors = pos + neg
    if not all_factors:
        st.info("No significant feature contributions found.")
        return

    all_factors.sort(key=lambda x: abs(x["contribution"]))

    features = [f["feature"] for f in all_factors]
    contributions = [f["contribution"] for f in all_factors]
    colors = [COLORS["success"] if c > 0 else COLORS["danger"] for c in contributions]

    fig = go.Figure(
        go.Bar(
            x=contributions,
            y=features,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<extra></extra>",
        )
    )

    fig.update_layout(
        height=min(300, max(150, len(features) * 40)),
        margin=dict(l=10, r=10, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["border"],
            color=COLORS["text_secondary"],
            title="Impact on Price",
            zerolinecolor=COLORS["text_secondary"],
        ),
        yaxis=dict(
            showgrid=False, color=COLORS["text_primary"], tickfont=dict(size=12)
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
