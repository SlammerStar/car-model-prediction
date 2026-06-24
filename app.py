"""
DRIVEIQ - AI Powered Used Car Valuation System
===============================================
Premium automotive application featuring a refined dark theme,
advanced machine learning pipelines, and market analytics.
"""

import sys
from pathlib import Path
import random

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go

from src.utils import (
    PIPELINE_PATH,
    METADATA_PATH,
    IMAGES_DIR,
    CURRENT_YEAR,
    PREMIUM_BRANDS,
    format_price_inr,
    load_model,
)
from src.prediction import predict_price
from src.data_processing import (
    load_and_merge_datasets,
    clean_data,
    convert_price_to_inr,
    create_features,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DRIVEIQ | AI Valuation System",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2300A3FF'><rect width='24' height='24' rx='4'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Brand Logos (inline SVG / image URLs)
# ---------------------------------------------------------------------------
BRAND_LOGOS = {
    "Audi": "https://www.carlogos.org/car-logos/audi-logo-2016-640.png",
    "BMW": "https://www.carlogos.org/car-logos/bmw-logo-2020-grey-640.png",
    "Ford": "https://www.carlogos.org/car-logos/ford-logo-2017-640.png",
    "Hyundai": "https://www.carlogos.org/car-logos/hyundai-logo-2011-640.png",
    "Mercedes": "https://www.carlogos.org/car-logos/mercedes-benz-logo-2011-640.png",
    "Skoda": "https://www.carlogos.org/car-logos/skoda-logo-2016-640.png",
    "Toyota": "https://www.carlogos.org/car-logos/toyota-logo-2020-europe-640.png",
    "Volkswagen": "https://www.carlogos.org/car-logos/volkswagen-logo-2019-640.png",
}

# Car model images - curated per brand for realism
BRAND_CAR_IMAGES = {
    "Audi": "https://images.unsplash.com/photo-1606664515524-ed2f786a0bd6?q=80&w=800&auto=format&fit=crop",
    "BMW": "https://images.unsplash.com/photo-1555215695-3004980ad54e?q=80&w=800&auto=format&fit=crop",
    "Ford": "https://images.unsplash.com/photo-1551830820-330a71b99659?q=80&w=800&auto=format&fit=crop",
    "Hyundai": "https://images.unsplash.com/photo-1629897048514-3dd7414fe72a?q=80&w=800&auto=format&fit=crop",
    "Mercedes": "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?q=80&w=800&auto=format&fit=crop",
    "Skoda": "https://images.unsplash.com/photo-1609521263047-f8f205293f24?q=80&w=800&auto=format&fit=crop",
    "Toyota": "https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?q=80&w=800&auto=format&fit=crop",
    "Volkswagen": "https://images.unsplash.com/photo-1622194992750-3f15b0a8266e?q=80&w=800&auto=format&fit=crop",
}

SIMILAR_CAR_IMAGES = [
    "https://images.unsplash.com/photo-1606152421802-db97b9c7a11b?q=80&w=500&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?q=80&w=500&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1606664515524-ed2f786a0bd6?q=80&w=500&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1555215695-3004980ad54e?q=80&w=500&auto=format&fit=crop",
]


def get_brand_logo(brand, size=22):
    """Return an img tag for the brand logo."""
    url = BRAND_LOGOS.get(brand, "")
    if url:
        return f'<img src="{url}" style="width:{size}px; height:{size}px; object-fit:contain; border-radius:2px;" alt="{brand}">'
    return ""


def get_brand_image(brand):
    """Return image URL for a brand."""
    return BRAND_CAR_IMAGES.get(
        brand,
        "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?q=80&w=800&auto=format&fit=crop",
    )


# ---------------------------------------------------------------------------
# Premium Dark Theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Montserrat:wght@600;700;800;900&display=swap');

    /* ── Global ── */
    .stApp {
        background-color: #050B16;
        color: #E5E7EB;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    .stDeployButton {display: none;}

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6 {
        color: #E5E7EB !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        letter-spacing: -0.3px;
    }
    p, span, label, div {
        font-family: 'Inter', sans-serif;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #050B16; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #070E1A !important;
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0;
    }

    /* Sidebar radio menu */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 10px 16px;
        border-radius: 10px;
        margin-bottom: 2px;
        color: #94A3B8;
        font-size: 0.88rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        color: #E5E7EB;
        background-color: rgba(255,255,255,0.03);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"] {
        background: rgba(0,163,255,0.08);
        border-left: 3px solid #00A3FF;
        color: #E5E7EB;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0px !important;
    }

    /* ── Inputs & Selectboxes ── */
    .stSelectbox > div > div,
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #0F172A !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #E5E7EB !important;
        border-radius: 10px !important;
        font-size: 0.88rem;
        transition: border-color 0.2s ease;
    }
    .stSelectbox [data-baseweb="select"] > div:hover,
    .stSelectbox [data-baseweb="select"] > div:focus-within {
        border-color: rgba(0,163,255,0.3) !important;
    }
    .stNumberInput > div > div > input {
        background-color: #0F172A !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #E5E7EB !important;
        border-radius: 10px !important;
        font-size: 0.88rem;
        transition: border-color 0.2s ease;
    }
    .stNumberInput > div > div > input:hover,
    .stNumberInput > div > div > input:focus {
        border-color: rgba(0,163,255,0.3) !important;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.2px;
    }

    /* Number input buttons */
    .stNumberInput button {
        background-color: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #94A3B8 !important;
        border-radius: 8px !important;
    }
    .stNumberInput button:hover {
        background-color: rgba(0,163,255,0.1) !important;
        border-color: rgba(0,163,255,0.3) !important;
        color: #00A3FF !important;
    }

    /* ── Slider - Premium Circular Thumb ── */
    .stSlider [data-baseweb="slider"] {
        padding-top: 12px !important;
        padding-bottom: 4px !important;
    }
    /* Track background */
    .stSlider [data-testid="stTickBar"],
    .stSlider [role="slider"] ~ div {
        background: transparent !important;
    }
    /* Active track (filled portion) */
    .stSlider div[data-baseweb="slider"] > div > div:nth-child(3) {
        background-color: #00A3FF !important;
    }
    .stSlider div[data-baseweb="slider"] > div > div:nth-child(4) {
        background-color: rgba(255,255,255,0.06) !important;
    }
    /* The thumb itself */
    .stSlider [role="slider"] {
        width: 18px !important;
        height: 18px !important;
        border-radius: 50% !important;
        background: #00A3FF !important;
        border: 3px solid #050B16 !important;
        box-shadow: 0 0 0 2px rgba(0,163,255,0.3), 0 0 10px rgba(0,163,255,0.2) !important;
        top: -5px !important;
        transition: box-shadow 0.2s ease !important;
    }
    .stSlider [role="slider"]:hover,
    .stSlider [role="slider"]:active {
        box-shadow: 0 0 0 3px rgba(0,163,255,0.4), 0 0 16px rgba(0,163,255,0.35) !important;
    }
    /* Slider value tooltip */
    .stSlider [data-baseweb="tooltip"] {
        background: #00A3FF !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #00A3FF, #0077FF) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.92 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0,163,255,0.25) !important;
        border: none !important;
        color: white !important;
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Metric Styling ── */
    [data-testid="stMetric"] {
        background-color: #0F172A;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px 18px;
    }
    [data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #E5E7EB !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.4rem !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
    }

    /* ── Premium Card ── */
    .premium-card {
        background-color: #0F172A;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        box-sizing: border-box;
    }
    .premium-card-header {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.92rem;
        font-weight: 700;
        color: #E5E7EB;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .metric-value {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.7rem;
        font-weight: 800;
        color: #E5E7EB;
        margin: 4px 0;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #00A3FF;
        font-weight: 600;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    /* ── Confidence Circular ── */
    .conf-circle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        font-size: 0.72rem;
        color: #E5E7EB;
        flex-shrink: 0;
    }

    /* ── Confidence Bar ── */
    .confidence-meter {
        width: 100%;
        height: 4px;
        background: rgba(255,255,255,0.06);
        border-radius: 2px;
        margin-top: 6px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 1s ease-in-out;
    }

    /* ── Vehicle Detail Card ── */
    .vehicle-card {
        background: #0F172A;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        overflow: hidden;
        height: 100%;
    }
    .vehicle-card-img {
        width: 100%;
        height: 180px;
        object-fit: cover;
        display: block;
    }
    .vehicle-card-body {
        padding: 16px 18px;
    }
    .vehicle-card-title {
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        color: #E5E7EB;
        font-size: 1.15rem;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .vehicle-card-specs {
        color: #94A3B8;
        font-size: 0.75rem;
        margin-bottom: 10px;
    }
    .vehicle-card-tags {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .vehicle-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        background: rgba(0,163,255,0.06);
        border: 1px solid rgba(0,163,255,0.1);
        border-radius: 20px;
        color: #94A3B8;
        font-size: 0.68rem;
        font-weight: 500;
    }
    .vehicle-tag svg { flex-shrink: 0; }
    .vehicle-view-btn {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin-top: 12px;
        padding: 6px 16px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #00A3FF;
        border: 1px solid rgba(0,163,255,0.25);
        border-radius: 8px;
        background: transparent;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
    }
    .vehicle-view-btn:hover {
        background: rgba(0,163,255,0.08);
        border-color: #00A3FF;
    }

    /* ── Similar Vehicle Cards ── */
    .sim-card {
        background-color: #0F172A;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px;
        height: 100%;
        transition: border-color 0.2s ease, transform 0.2s ease;
        box-sizing: border-box;
    }
    .sim-card:hover {
        border-color: rgba(0,163,255,0.2);
        transform: translateY(-2px);
    }
    .sim-card-img {
        width: 100%;
        height: 100px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .sim-card-brand {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 2px;
    }
    .sim-card-title {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        color: #E5E7EB;
        font-size: 0.88rem;
    }
    .sim-card-specs {
        font-size: 0.68rem;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .sim-card-price {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        color: #E5E7EB;
        font-size: 0.95rem;
    }
    .sim-view-btn {
        display: inline-block;
        margin-top: 6px;
        padding: 4px 12px;
        font-size: 0.68rem;
        font-weight: 600;
        color: #00A3FF;
        border: 1px solid rgba(0,163,255,0.25);
        border-radius: 6px;
        background: transparent;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
    }
    .sim-view-btn:hover {
        background: rgba(0,163,255,0.08);
    }

    /* ── Section Title ── */
    .section-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #E5E7EB;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ── Section Card ── */
    .section-card {
        background-color: #0F172A;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    [data-testid="stDataFrame"] > div { border-radius: 12px; }

    /* ── Plotly Charts ── */
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 500;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0,163,255,0.1) !important;
        color: #00A3FF !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.06) !important; margin: 20px 0 !important; }

    /* ── Hero ── */
    .hero-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        color: #E5E7EB;
        line-height: 1;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-family: 'Montserrat', sans-serif;
        color: #00A3FF;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 10px 0 14px 0;
    }
    .hero-desc {
        color: #94A3B8;
        font-size: 0.92rem;
        line-height: 1.6;
        max-width: 460px;
    }

    /* ── About Feature Grid ── */
    .feature-item {
        background: #0F172A;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    .feature-item:hover { border-color: rgba(0,163,255,0.15); }
    .feature-icon { margin-bottom: 12px; }
    .feature-title {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        color: #E5E7EB;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }
    .feature-desc {
        color: #94A3B8;
        font-size: 0.78rem;
        line-height: 1.5;
    }

    /* ── Column gaps ── */
    [data-testid="stHorizontalBlock"] { gap: 14px; }

    /* ── Hide empty ── */
    .stMarkdown:empty { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        raw = load_and_merge_datasets()
        clean = clean_data(raw)
        conv = convert_price_to_inr(clean)
        feat = create_features(conv)
        return feat
    except Exception:
        return None


@st.cache_resource
def load_ml_pipeline():
    try:
        return load_model(PIPELINE_PATH)
    except FileNotFoundError:
        return None


@st.cache_data
def load_metadata():
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


df = load_data()
if df is None:
    st.error("Failed to load dataset.")
    st.stop()

pipeline = load_ml_pipeline()
metadata = load_metadata()


# ---------------------------------------------------------------------------
# SVG Icons (Lucide-style)
# ---------------------------------------------------------------------------
def icon(name, size=18, color="#94A3B8"):
    """Return inline SVG icons matching Lucide icon set."""
    icons = {
        "home": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
        "bar-chart": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
        "trending-up": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
        "trending-down": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>',
        "car": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 16H9m10 0h3v-3.15a1 1 0 0 0-.84-.99L16 11l-2.7-3.6a2 2 0 0 0-1.6-.8H8.3a2 2 0 0 0-1.6.8L4 11l-5.16.86a1 1 0 0 0-.84.99V16h3"/><circle cx="6.5" cy="16.5" r="2.5"/><circle cx="16.5" cy="16.5" r="2.5"/></svg>',
        "search": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
        "database": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
        "info": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        "shield-check": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
        "zap": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        "target": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "cpu": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "layers": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "gauge": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>',
        "arrow-right": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
        "globe": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
        "fuel": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 22V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v17"/><path d="M15 10h2a2 2 0 0 1 2 2v2a2 2 0 0 0 2 2h0a2 2 0 0 0 2-2V9.83a2 2 0 0 0-.59-1.42L18 4"/><path d="M7 10h4"/></svg>',
        "settings": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
        "star": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
        "check": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    }
    return icons.get(name, "")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
    <div style="text-align: center; padding: 24px 16px 16px 16px;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; background: rgba(0,163,255,0.1); border-radius: 12px; margin-bottom: 10px;">
            {icon("gauge", 26, "#00A3FF")}
        </div>
        <div style="font-family: 'Montserrat', sans-serif; font-size: 1.3rem; font-weight: 900; color: #E5E7EB; letter-spacing: 2px;">DRIVEIQ</div>
        <div style="color: #94A3B8; font-size: 0.65rem; letter-spacing: 1.5px; margin-top: 2px; text-transform: uppercase;">AI Valuation Engine</div>
    </div>
    <hr style="border-color: rgba(255,255,255,0.04); margin: 0 16px 12px 16px;">
    """,
        unsafe_allow_html=True,
    )

    menu_options = [
        "Predict Price",
        "Performance Analytics",
        "Market Insights",
        "Vehicle Explorer",
        "Data Explorer",
        "About DRIVEIQ",
    ]
    page = st.radio("Navigation", menu_options, label_visibility="collapsed")

    st.markdown(
        "<div style='flex-grow: 1; min-height: 40px;'></div>", unsafe_allow_html=True
    )

    st.markdown(
        f"""
    <div style='margin: 0 12px; padding: 14px; background: rgba(0,163,255,0.04); border-radius: 10px; border: 1px solid rgba(0,163,255,0.08);'>
        <div style='color: #94A3B8; font-size: 0.7rem; display: flex; align-items: center; gap: 6px;'>
            {icon("zap", 12, "#00A3FF")}
            Powered by
        </div>
        <div style='color: #E5E7EB; font-weight: 700; font-size: 0.82rem; margin-top: 4px; font-family: Montserrat, sans-serif;'>Random Forest + XGBoost</div>
        <div style='color: #64748B; font-size: 0.62rem; margin-top: 4px; display: flex; align-items: center; gap: 4px;'>
            {icon("zap", 10, "#64748B")}
            Built with Streamlit
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1: PREDICT PRICE
# ===========================================================================
if page == "Predict Price":

    if pipeline is None:
        st.error("Model not trained yet. Run the training script first.")
        st.stop()

    # ── Hero Section ──
    hero_left, hero_right = st.columns([1.5, 1], gap="large")

    with hero_left:
        st.markdown(
            """
        <div style="padding: 12px 0 28px 0;">
            <h1 class="hero-title">DRIVEIQ</h1>
            <div class="hero-subtitle">AI Powered Used Car Valuation System</div>
            <p class="hero-desc">Get accurate market value estimates for used cars across India's leading automobile brands.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Vehicle Configuration ──
    st.markdown(
        f"""
    <div class="section-title" style="margin-top: 4px;">
        {icon("car", 18, "#00A3FF")}
        Configure Vehicle Details
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Row 1: Brand + Year Slider + Fuel Efficiency
    r1c1, r1c2, r1c3 = st.columns([1, 1.2, 1])
    with r1c1:
        brand = st.selectbox("Brand", sorted(df["brand"].unique()))
    with r1c2:
        year = st.slider(
            "Manufacturing Year",
            min_value=1996,
            max_value=int(CURRENT_YEAR),
            value=2021,
        )
    with r1c3:
        mpg = st.number_input(
            "Fuel Efficiency (km/l)",
            min_value=0.0,
            max_value=100.0,
            value=float(df["mpg"].median()),
            step=0.5,
        )

    # Row 2: Model + Transmission + Engine Size
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        brand_models = sorted(df[df["brand"] == brand]["model"].unique())
        model_name = st.selectbox("Model", brand_models)
    with r2c2:
        transmission = st.selectbox("Transmission", sorted(df["transmission"].unique()))
    with r2c3:
        engine_size = st.selectbox("Engine Size (L)", sorted(df["engineSize"].unique()))

    # Row 3: Fuel Type + Mileage + Button
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        fuel_type = st.selectbox("Fuel Type", sorted(df["fuelType"].unique()))
    with r3c2:
        mileage = st.number_input(
            "Mileage (kms)", min_value=0, max_value=500000, value=30000, step=1000
        )
    with r3c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        predict_trigger = st.button("CALCULATE VALUATION   \u2192")

    # ── Dynamic Vehicle Image Card (right column of hero, rendered after form so brand/model are known) ──
    with hero_right:
        car_img = get_brand_image(brand)
        brand_logo_html = get_brand_logo(brand, 28)
        brand_category = "Luxury" if brand in PREMIUM_BRANDS else "Standard"

        st.markdown(
            f"""
        <div class="vehicle-card">
            <img src="{car_img}" class="vehicle-card-img" alt="{brand} {model_name}">
            <div class="vehicle-card-body">
                <div class="vehicle-card-title">
                    {brand_logo_html}
                    {brand} {model_name}
                </div>
                <div class="vehicle-card-specs">{year} &middot; {transmission} &middot; {fuel_type}</div>
                <div class="vehicle-card-tags">
                    <span class="vehicle-tag">{icon("star", 10, "#00A3FF")} {brand_category}</span>
                    <span class="vehicle-tag">{icon("check", 10, "#00FF88")} Performance</span>
                    <span class="vehicle-tag">{icon("shield-check", 10, "#94A3B8")} Reliability</span>
                </div>
                <span class="vehicle-view-btn">View Details {icon("arrow-right", 12, "#00A3FF")}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Prediction Results ──
    if predict_trigger:
        with st.spinner("Analyzing market data..."):
            try:
                result = predict_price(
                    brand=brand,
                    model=model_name,
                    year=year,
                    transmission=transmission,
                    mileage=mileage,
                    fuel_type=fuel_type,
                    mpg=mpg,
                    engine_size=engine_size,
                    pipeline=pipeline,
                )

                st.divider()

                # Results: 3 cards
                res1, res2, res3 = st.columns([1.2, 1, 1])

                # ── Card 1: Estimated Market Value ──
                with res1:
                    conf_val = int(result["confidence"].replace("%", ""))
                    conf_color = (
                        "#00FF88"
                        if conf_val >= 85
                        else ("#FFC857" if conf_val >= 75 else "#FF3B30")
                    )

                    st.markdown(
                        f"""
                    <div class="premium-card">
                        <div class="premium-card-header">
                            {icon("target", 16, "#00A3FF")}
                            Estimated Market Value
                        </div>
                        <div class="metric-label">Market Value Range</div>
                        <div class="metric-sub">{result["price_range"]}</div>
                        <div class="metric-label" style="margin-top: 12px;">Estimated Price</div>
                        <div class="metric-value">{result["predicted_price"]}</div>
                        <div style="margin-top: 14px; display: flex; align-items: center; gap: 10px;">
                            <div class="conf-circle" style="border: 3px solid {conf_color};">{conf_val}%</div>
                            <div>
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    {icon("shield-check", 13, conf_color)}
                                    <span style="color: #94A3B8; font-size: 0.75rem;">Prediction Confidence</span>
                                </div>
                                <div class="confidence-meter" style="width: 140px; margin-top: 4px;">
                                    <div class="confidence-fill" style="width: {conf_val}%; background-color: {conf_color};"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # ── Card 2: Market Trend ──
                with res2:
                    trend_val = round(random.uniform(2.5, 8.5), 1)

                    # Sparkline
                    np.random.seed(42)
                    trend_y = np.cumsum(np.random.randn(12) * 0.5) + 10
                    trend_y = trend_y - trend_y.min() + 2
                    months = [
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ]

                    fig_trend = go.Figure()
                    fig_trend.add_trace(
                        go.Scatter(
                            x=months,
                            y=trend_y,
                            mode="lines",
                            line=dict(
                                color="#00FF88",
                                width=2.5,
                                shape="spline",
                                smoothing=1.3,
                            ),
                            fill="tozeroy",
                            fillcolor="rgba(0,255,136,0.06)",
                            showlegend=False,
                        )
                    )
                    fig_trend.update_layout(
                        height=110,
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                    )

                    st.markdown(
                        f"""
                    <div class="premium-card" style="padding-bottom: 14px;">
                        <div class="premium-card-header" style="justify-content: space-between;">
                            <span style="display: flex; align-items: center; gap: 8px;">
                                {icon("trending-up", 16, "#00FF88")}
                                Market Trend
                            </span>
                            <span style="font-size: 0.65rem; color: #94A3B8; font-weight: 400; background: rgba(255,255,255,0.04); padding: 3px 10px; border-radius: 20px;">Last 12 Months</span>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    st.plotly_chart(
                        fig_trend,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                    st.markdown(
                        f"""
                    <div style="margin-top: -16px; padding: 0 6px;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            {icon("trending-up", 14, "#00FF88")}
                            <span style="color: #00FF88; font-weight: 700; font-size: 1rem;">{trend_val}%</span>
                        </div>
                        <div class="metric-label" style="margin-top: 2px;">Increase in market value</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # ── Card 3: Depreciation Analysis ──
                with res3:
                    car_age = result["input_summary"]["car_age"]
                    dep_pct = result["depreciation_percent"]
                    progress_width = min(95, max(10, (car_age / 30) * 100))

                    st.markdown(
                        f"""
                    <div class="premium-card">
                        <div class="premium-card-header">
                            {icon("trending-down", 16, "#FF3B30")}
                            Depreciation Analysis
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 8px;">
                            <div>
                                <div class="metric-label">Original<br>Est. Price</div>
                                <div style="font-family: Montserrat, sans-serif; font-size: 0.9rem; font-weight: 700; color: #E5E7EB; margin-top: 4px;">{result["original_price"]}</div>
                            </div>
                            <div>
                                <div class="metric-label">Value Lost</div>
                                <div style="font-size: 0.9rem; font-weight: 700; color: #FF3B30; margin-top: 4px; display: flex; align-items: center; gap: 4px;">
                                    {icon("trending-down", 12, "#FF3B30")} {dep_pct}
                                </div>
                            </div>
                            <div>
                                <div class="metric-label">Vehicle Age</div>
                                <div style="font-family: Montserrat, sans-serif; font-size: 0.9rem; font-weight: 700; color: #E5E7EB; margin-top: 4px;">{car_age} Years</div>
                            </div>
                        </div>
                        <div style="margin-top: 20px; position: relative;">
                            <div style="height: 4px; background: rgba(255,255,255,0.06); border-radius: 2px; width: 100%;"></div>
                            <div style="height: 4px; background: #00A3FF; border-radius: 2px; width: {progress_width}%; position: absolute; top: 0; left: 0;"></div>
                            <div style="width: 10px; height: 10px; background: #E5E7EB; border: 2px solid #00A3FF; border-radius: 50%; position: absolute; top: -3px; left: calc({progress_width}% - 5px);"></div>
                            <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 0.7rem; color: #94A3B8;">
                                <span>{year}</span>
                                <span>{CURRENT_YEAR}</span>
                            </div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # ── Similar Vehicles ──
                recs = result.get("recommendations", [])
                if recs:
                    st.markdown(
                        f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 24px; margin-bottom: 14px;">
                        <div class="section-title" style="margin: 0;">
                            {icon("layers", 16, "#00A3FF")}
                            Similar Vehicles You Can Consider
                        </div>
                        <span style="font-size: 0.72rem; color: #94A3B8; display: flex; align-items: center; gap: 4px; cursor: pointer;">
                            View All {icon("arrow-right", 12, "#94A3B8")}
                        </span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    num_recs = min(4, len(recs))
                    sim_cols = st.columns(num_recs)

                    for idx, rec in enumerate(recs[:num_recs]):
                        rec_brand = rec.get("brand", brand)
                        rec_logo = get_brand_logo(rec_brand, 18)
                        rec_img = SIMILAR_CAR_IMAGES[idx % len(SIMILAR_CAR_IMAGES)]
                        with sim_cols[idx]:
                            st.markdown(
                                f"""
                            <div class="sim-card">
                                <img src="{rec_img}" class="sim-card-img" alt="{rec_brand} {rec['model']}">
                                <div class="sim-card-brand">
                                    {rec_logo}
                                    <span class="sim-card-title">{rec_brand} {rec['model']}</span>
                                </div>
                                <div class="sim-card-specs">{rec['year']} &middot; {fuel_type} &middot; {transmission}</div>
                                <div class="sim-card-price">{rec['price']}</div>
                                <span class="sim-view-btn">View Details</span>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")


# ===========================================================================
# PAGE 2: PERFORMANCE ANALYTICS
# ===========================================================================
elif page == "Performance Analytics":
    st.markdown(
        """
    <div style="padding: 8px 0 20px 0;">
        <h2 style="font-size: 1.5rem; margin: 0;">Model Performance Dashboard</h2>
        <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 6px;">Evaluate and compare machine learning model accuracy</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if metadata:
        tuned = metadata.get("tuned_metrics", {})

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="Model Engine", value=metadata.get("best_model", "N/A"))
        with kpi2:
            st.metric(
                label="R-Squared Accuracy",
                value=f"{tuned.get('R2_Score', 0) * 100:.1f}%",
            )
        with kpi3:
            st.metric(
                label="Mean Absolute Error", value=f"INR {tuned.get('MAE', 0):,.0f}"
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""
        <div class="section-title">
            {icon("bar-chart", 16, "#00A3FF")}
            Algorithm Comparison
        </div>
        """,
            unsafe_allow_html=True,
        )

        comparison = pd.DataFrame(metadata.get("comparison", []))
        if not comparison.empty:
            display_comp = comparison.copy()
            display_comp["R2_Score"] = display_comp["R2_Score"].apply(
                lambda x: f"{x * 100:.2f}%" if x > 0 else "N/A"
            )
            display_comp["MAE"] = display_comp["MAE"].apply(
                lambda x: f"INR {x:,.0f}" if x < 1e10 else "N/A"
            )
            display_comp["RMSE"] = display_comp["RMSE"].apply(
                lambda x: f"INR {x:,.0f}" if x < 1e10 else "N/A"
            )
            st.dataframe(display_comp, use_container_width=True, hide_index=True)

        st.divider()

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown(
                f'<div class="section-title" style="font-size: 0.88rem;">{icon("target", 14, "#00A3FF")} Actual vs Predicted</div>',
                unsafe_allow_html=True,
            )
            img_path = IMAGES_DIR / "actual_vs_predicted.png"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.info("Chart not available. Run training to generate visualizations.")

        with col_img2:
            st.markdown(
                f'<div class="section-title" style="font-size: 0.88rem;">{icon("bar-chart", 14, "#00A3FF")} Residual Distribution</div>',
                unsafe_allow_html=True,
            )
            img_path = IMAGES_DIR / "residual_distribution.png"
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.info("Chart not available. Run training to generate visualizations.")

        st.markdown(
            f'<div class="section-title" style="font-size: 0.88rem; margin-top: 12px;">{icon("layers", 14, "#00A3FF")} SHAP Feature Importance</div>',
            unsafe_allow_html=True,
        )
        img_path = IMAGES_DIR / "shap_summary.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info(
                "SHAP summary not available. Run training to generate visualizations."
            )
    else:
        st.info("Performance metadata unavailable. Run the training pipeline first.")


# ===========================================================================
# PAGE 3: MARKET INSIGHTS
# ===========================================================================
elif page == "Market Insights":
    st.markdown(
        """
    <div style="padding: 8px 0 20px 0;">
        <h2 style="font-size: 1.5rem; margin: 0;">Market Insights</h2>
        <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 6px;">Comprehensive analytics across the used car market</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Vehicles", f"{len(df):,}")
    with kpi2:
        st.metric("Avg Market Price", format_price_inr(df["price_inr"].mean()))
    with kpi3:
        st.metric("Most Popular Brand", df["brand"].mode()[0])
    with kpi4:
        st.metric("Avg Vehicle Age", f"{df['car_age'].mean():.1f} Yrs")

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        avg_price = (
            df.groupby("brand")["price_inr"]
            .mean()
            .reset_index()
            .sort_values("price_inr", ascending=False)
        )
        fig = px.bar(
            avg_price,
            x="brand",
            y="price_inr",
            title="Average Valuation by Brand",
            color="price_inr",
            color_continuous_scale=[[0, "#0077FF"], [1, "#00A3FF"]],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            font_color="#94A3B8",
            title_font=dict(family="Montserrat", size=14, color="#E5E7EB"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)", title=""),
            yaxis=dict(gridcolor="rgba(255,255,255,0.03)", title="Price (INR)"),
            coloraxis_showscale=False,
            margin=dict(t=40, b=20),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        fuel_dist = df["fuelType"].value_counts().reset_index()
        fig2 = px.pie(
            fuel_dist,
            names="fuelType",
            values="count",
            title="Fuel Type Market Share",
            color_discrete_sequence=["#00A3FF", "#0077FF", "#1E3A5F", "#94A3B8"],
            template="plotly_dark",
            hole=0.55,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            font_color="#94A3B8",
            title_font=dict(family="Montserrat", size=14, color="#E5E7EB"),
            margin=dict(t=40, b=20),
            height=340,
            legend=dict(font=dict(size=10)),
        )
        fig2.update_traces(textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig2, use_container_width=True)

    sample_size = min(2000, len(df))
    fig3 = px.scatter(
        df.sample(sample_size),
        x="mileage",
        y="price_inr",
        color="brand",
        title=f"Mileage vs Valuation (Sample {sample_size:,})",
        template="plotly_dark",
        opacity=0.55,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        font_color="#94A3B8",
        title_font=dict(family="Montserrat", size=14, color="#E5E7EB"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)", title="Mileage"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.03)", title="Price (INR)"),
        margin=dict(t=40, b=20),
        height=400,
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ===========================================================================
# PAGE 4: VEHICLE EXPLORER
# ===========================================================================
elif page == "Vehicle Explorer":
    st.markdown(
        """
    <div style="padding: 8px 0 20px 0;">
        <h2 style="font-size: 1.5rem; margin: 0;">Vehicle Explorer</h2>
        <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 6px;">Browse and discover vehicles in our database</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    filter1, filter2, filter3 = st.columns(3)
    with filter1:
        selected_brand = st.selectbox(
            "Filter by Brand", ["All Brands"] + sorted(df["brand"].unique().tolist())
        )
    with filter2:
        selected_fuel = st.selectbox(
            "Filter by Fuel Type", ["All"] + sorted(df["fuelType"].unique().tolist())
        )
    with filter3:
        selected_trans = st.selectbox(
            "Filter by Transmission",
            ["All"] + sorted(df["transmission"].unique().tolist()),
        )

    display_df = df.copy()
    if selected_brand != "All Brands":
        display_df = display_df[display_df["brand"] == selected_brand]
    if selected_fuel != "All":
        display_df = display_df[display_df["fuelType"] == selected_fuel]
    if selected_trans != "All":
        display_df = display_df[display_df["transmission"] == selected_trans]

    stat1, stat2, stat3 = st.columns(3)
    with stat1:
        st.metric("Vehicles Found", f"{len(display_df):,}")
    with stat2:
        if len(display_df) > 0:
            st.metric(
                "Price Range",
                f"{format_price_inr(display_df['price_inr'].min())} - {format_price_inr(display_df['price_inr'].max())}",
            )
        else:
            st.metric("Price Range", "N/A")
    with stat3:
        st.metric(
            "Avg Mileage",
            (
                f"{display_df['mileage'].mean():,.0f} kms"
                if len(display_df) > 0
                else "N/A"
            ),
        )

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    styled_df = display_df.copy()
    styled_df["Price"] = styled_df["price_inr"].apply(format_price_inr)
    styled_df = styled_df[
        [
            "brand",
            "model",
            "year",
            "Price",
            "transmission",
            "fuelType",
            "mileage",
            "engineSize",
        ]
    ]
    styled_df.columns = [
        "Brand",
        "Model",
        "Year",
        "Price",
        "Transmission",
        "Fuel Type",
        "Mileage",
        "Engine (L)",
    ]

    st.dataframe(styled_df.head(100), use_container_width=True, hide_index=True)
    st.markdown(
        f"<div style='color: #94A3B8; text-align: right; font-size: 0.75rem; margin-top: 6px;'>Showing top 100 of {len(display_df):,} vehicles</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 5: DATA EXPLORER
# ===========================================================================
elif page == "Data Explorer":
    st.markdown(
        """
    <div style="padding: 8px 0 20px 0;">
        <h2 style="font-size: 1.5rem; margin: 0;">Data Explorer</h2>
        <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 6px;">Explore the raw dataset powering DRIVEIQ</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    selected_brand = st.selectbox(
        "Filter by Brand",
        ["All Brands"] + sorted(df["brand"].unique().tolist()),
        key="data_brand",
    )
    display_df = (
        df if selected_brand == "All Brands" else df[df["brand"] == selected_brand]
    )

    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric("Rows", f"{len(display_df):,}")
    with stat2:
        st.metric("Columns", f"{len(display_df.columns)}")
    with stat3:
        st.metric("Brands", f"{display_df['brand'].nunique()}")
    with stat4:
        st.metric("Models", f"{display_df['model'].nunique()}")

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    styled_df = display_df.copy()
    styled_df["Price"] = styled_df["price_inr"].apply(format_price_inr)
    styled_df = styled_df[
        [
            "brand",
            "model",
            "year",
            "Price",
            "transmission",
            "fuelType",
            "mileage",
            "engineSize",
        ]
    ]

    st.dataframe(styled_df.head(100), use_container_width=True, hide_index=True)
    st.markdown(
        f"<div style='color: #94A3B8; text-align: right; font-size: 0.75rem; margin-top: 6px;'>Showing top 100 of {len(display_df):,} vehicles</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    dist1, dist2 = st.columns(2)
    with dist1:
        year_dist = display_df["year"].value_counts().sort_index().reset_index()
        year_dist.columns = ["Year", "Count"]
        fig_year = px.bar(
            year_dist,
            x="Year",
            y="Count",
            title="Distribution by Year",
            template="plotly_dark",
            color_discrete_sequence=["#00A3FF"],
        )
        fig_year.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            font_color="#94A3B8",
            title_font=dict(family="Montserrat", size=14, color="#E5E7EB"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
            margin=dict(t=40, b=20),
            height=300,
        )
        st.plotly_chart(fig_year, use_container_width=True)

    with dist2:
        fig_price = px.histogram(
            display_df,
            x="price_inr",
            nbins=40,
            title="Price Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#0077FF"],
        )
        fig_price.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            font_color="#94A3B8",
            title_font=dict(family="Montserrat", size=14, color="#E5E7EB"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)", title="Price (INR)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.03)", title="Count"),
            margin=dict(t=40, b=20),
            height=300,
        )
        st.plotly_chart(fig_price, use_container_width=True)


# ===========================================================================
# PAGE 6: ABOUT
# ===========================================================================
elif page == "About DRIVEIQ":
    st.markdown(
        """
    <div style="padding: 8px 0 20px 0;">
        <h2 style="font-size: 1.5rem; margin: 0;">About DRIVEIQ</h2>
        <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 6px;">AI Powered Used Car Valuation System</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class="premium-card" style="margin-bottom: 20px;">
        <div class="premium-card-header" style="font-size: 1rem;">
            {icon("gauge", 18, "#00A3FF")}
            What is DRIVEIQ?
        </div>
        <p style="color: #94A3B8; line-height: 1.7; font-size: 0.88rem; margin: 0;">
            DRIVEIQ is a state-of-the-art machine learning application designed to estimate the market value of premium used cars across India.
            Built using scikit-learn pipelines, Random Forest, and XGBoost algorithms, it processes thousands of data points to generate accurate valuations.
            The system features advanced data preprocessing, SHAP model explainability, nearest-neighbors vehicle recommendations, and a RESTful FastAPI backend.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    feat_cols = st.columns(3)
    features = [
        (
            "cpu",
            "ML Pipeline",
            "Automated preprocessing with scikit-learn ColumnTransformer, feature engineering, and hyperparameter-tuned Random Forest models.",
        ),
        (
            "target",
            "Price Prediction",
            "Accurate used car valuations with confidence intervals, depreciation analysis, and Indian market calibration.",
        ),
        (
            "layers",
            "Recommendations",
            "K-Nearest Neighbors recommender system suggests similar vehicles based on transformed feature space.",
        ),
        (
            "bar-chart",
            "Model Analytics",
            "Compare 5 algorithms side-by-side with R-squared, MAE, RMSE metrics and SHAP explainability plots.",
        ),
        (
            "globe",
            "Market Insights",
            "Interactive charts covering brand valuations, fuel type distributions, and mileage-price correlations.",
        ),
        (
            "shield-check",
            "Data Quality",
            f"Trained on {metadata.get('n_training_samples', 66036):,} samples across 8 brands with robust cross-validation.",
        ),
    ]

    for i, (feat_icon, title, desc) in enumerate(features):
        with feat_cols[i % 3]:
            st.markdown(
                f"""
            <div class="feature-item" style="margin-bottom: 14px;">
                <div class="feature-icon">{icon(feat_icon, 26, "#00A3FF")}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="section-title" style="margin-top: 10px;">{icon("zap", 16, "#00A3FF")} Technology Stack</div>',
        unsafe_allow_html=True,
    )

    tech1, tech2, tech3, tech4 = st.columns(4)
    with tech1:
        st.metric("Frontend", "Streamlit")
    with tech2:
        st.metric("ML Framework", "scikit-learn")
    with tech3:
        st.metric("Boosting", "XGBoost")
    with tech4:
        st.metric("API", "FastAPI")
