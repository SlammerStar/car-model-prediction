# AI Valuation Dashboard Architecture

The Premium AI Valuation Dashboard is the frontend visualization layer for DRIVEIQ's Phase 6 Intelligence Engine. It processes the complex, structured dictionary returned by `predict_price()` and converts it into a rich, modern, and highly interactive user experience using Streamlit.

## Component Architecture

The dashboard logic is completely modularized in the `src/ui/` directory:

1. **`theme.py`**
   - Centralizes the color palette (`COLORS`) to maintain consistency.
   - Contains `load_dashboard_css()` which injects custom HTML/CSS styles for the cards, tags, text, and layout structuring.

2. **`valuation_cards.py`**
   - **Hero Card (`render_hero_card`)**: Displays the estimated market value, confidence gauge, position, and recommendation badge. Includes a computed "Valuation Score".
   - **Spec Card (`render_spec_card`)**: A read-only grid listing the configuration specs (engine, year, transmission, etc.).
   - **Similar Vehicles (`render_similar_vehicles`)**: Renders attractive responsive cards showcasing similar vehicles from the market data.

3. **`charts.py`**
   - **Price Gauge (`render_price_gauge`)**: A horizontal indicator tracking the predicted price inside the estimated variance bounds, colored with a gradient.
   - **Future Value Projection (`render_future_value_timeline`)**: A Plotly line chart projecting standard depreciation across 5 years to visualize future value retention.
   - **Feature Impact (`render_feature_contributions`)**: A Plotly horizontal bar chart displaying the positive and negative valuation factors (using translated explanation strings, not raw SHAP floats).

4. **`report_components.py`**
   - **AI Summary (`render_ai_summary`)**: A highlighted section rendering the AI-generated paragraph.
   - **Factor Cards (`render_factors_cards`)**: Two parallel vertical lists clearly separating the positive aspects from the negative aspects.
   - **Risk Assessment (`render_risk_assessment`)**: Visually indicates Ownership, Market, and Maintenance risks using colors (Green/Yellow/Red).

5. **`dashboard.py`**
   - The main orchestrator (`render_valuation_dashboard`).
   - Uses Streamlit Tabs to categorize the components into:
     - **AI Valuation Overview** (Hero, Summary, Factors, Gauge, Risks)
     - **Market Intelligence** (Feature Impact, Future Projection, Similar Vehicles)
     - **Vehicle Details** (Specs, Raw Payload)

6. **`export.py`**
   - Implements `ValuationPDF` leveraging `fpdf2` to build a professional, shareable PDF report containing the prediction, insights, and risks.

## Performance Optimization

The dashboard utilizes `Streamlit`'s native columns and layout primitives heavily. By moving CSS into a dedicated loader (`theme.py`), we reduce unnecessary re-rendering. 

## Integration

In `app.py`, the old hardcoded 3-card layout was stripped out. `app.py` simply imports `render_valuation_dashboard` and passes the response payload, keeping the application script clean and scalable.
