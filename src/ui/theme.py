import streamlit as st

COLORS = {
    "primary": "#00A3FF",
    "success": "#00FF88",
    "warning": "#FFC857",
    "danger": "#FF3B30",
    "background": "#0A0F1C",
    "card_bg": "#111827",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "border": "rgba(255,255,255,0.08)",
}


def load_dashboard_css():
    """Injects premium CSS for the valuation dashboard."""
    st.markdown(
        f"""
    <style>
    .val-hero-card {{
        background: linear-gradient(145deg, #111827 0%, #0F172A 100%);
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        padding: 24px;
        position: relative;
        overflow: hidden;
    }}
    .val-hero-glow {{
        position: absolute;
        top: -50px;
        right: -50px;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, rgba(0,163,255,0.15) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
    }}
    .val-price {{
        font-family: 'Montserrat', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: {COLORS["text_primary"]};
        margin: 10px 0 4px 0;
        letter-spacing: -1px;
    }}
    .val-range {{
        font-size: 0.95rem;
        color: {COLORS["text_secondary"]};
        font-weight: 500;
    }}
    .val-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .val-summary-card {{
        background: rgba(0,163,255,0.03);
        border-left: 4px solid {COLORS["primary"]};
        border-radius: 8px;
        padding: 20px;
        margin-top: 16px;
    }}
    .val-summary-text {{
        color: {COLORS["text_primary"]};
        font-size: 1.05rem;
        line-height: 1.6;
        font-weight: 400;
    }}
    .val-factor-list {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}
    .val-factor-item {{
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 12px 0;
        border-bottom: 1px solid {COLORS["border"]};
    }}
    .val-factor-item:last-child {{
        border-bottom: none;
    }}
    .val-factor-title {{
        color: {COLORS["text_primary"]};
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }}
    .val-factor-desc {{
        color: {COLORS["text_secondary"]};
        font-size: 0.85rem;
        line-height: 1.4;
    }}
    .val-spec-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
    }}
    .val-spec-item {{
        background: rgba(255,255,255,0.02);
        padding: 12px;
        border-radius: 8px;
    }}
    .val-spec-label {{
        color: {COLORS["text_secondary"]};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    .val-spec-value {{
        color: {COLORS["text_primary"]};
        font-weight: 600;
        font-size: 0.95rem;
    }}
    .val-similar-card {{
        background: {COLORS["card_bg"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 16px;
        transition: transform 0.2s, border-color 0.2s;
        cursor: pointer;
        height: 100%;
    }}
    .val-similar-card:hover {{
        transform: translateY(-2px);
        border-color: {COLORS["primary"]};
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
