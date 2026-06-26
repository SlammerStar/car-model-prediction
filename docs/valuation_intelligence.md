# AI Valuation Intelligence Engine

The AI Valuation Intelligence Engine is the core component of DRIVEIQ Phase 6 that translates raw machine learning predictions into business-friendly, human-readable valuation reports. It abstracts away the complexity of feature importance and raw probabilities, providing actionable insights for users.

## Architecture

The system is split into two primary components:

1. **Explanation Engine (`src/explanation_engine.py`)**
   - Implements a `BaseExplanationProvider` interface.
   - Provides a concrete `ShapExplanationProvider` that calculates the SHAP (SHapley Additive exPlanations) values for a given prediction.
   - **Business Translation Layer**: Maps raw mathematical SHAP values into localized, directional explanations. E.g., `car_age: -50000` is translated to "Vehicle Age reduces the market value."

2. **Valuation Intelligence Engine (`src/valuation_intelligence.py`)**
   - The central orchestrator that produces the complete `ValuationReport`.
   - **Confidence Estimator**: Computes a 0-100% confidence score by applying penalties for extreme vehicle age, configuration rarity, and low market liquidity.
   - **Dynamic Price Range**: Calculates the lower and upper bounds of a realistic asking price, expanding the variance if confidence is low or market stability is volatile.
   - **Market Position Classifier**: Maps the intrinsic brand value retention into categories: `Undervalued`, `Fairly Priced`, `Slightly Overpriced`, or `Overpriced`.
   - **Risk Assessor**: Translates specific features (`owner_risk_score`, `market_stability_score`, `age_based_depreciation`) into High/Medium/Low indicators for Ownership, Market, and Maintenance risks.
   - **AI Summary Generator**: Synthesizes the prediction, confidence, position, and top explanation factors into a natural language paragraph.

## Configuration

All business rules, translation dictionaries, and threshold values are maintained externally in `src/valuation_config.json`. This ensures that data scientists and business analysts can adjust market classification criteria without modifying the python source code.

## Output Format

The engine generates a structured dictionary attached to the prediction output:

```json
{
  "valuation_report": {
    "estimated_market_value": "₹ 5,00,000",
    "confidence": {
      "score": 85.5,
      "label": "High"
    },
    "market_position": "Fairly Priced",
    "recommendation": "Good Buy",
    "risk_assessment": {
      "Ownership Risk": "Low",
      "Market Risk": "Medium",
      "Maintenance Risk": "Low"
    },
    "ai_summary": "The Honda City has an estimated market value of ₹ 5,00,000 with high confidence. This vehicle is fairly aligned with current market expectations. The valuation is positively supported by its Engine Performance.",
    "explanation": {
      "major_positive_factors": [{"feature": "Engine Performance", "explanation": "..."}],
      "major_negative_factors": [{"feature": "Vehicle Age", "explanation": "..."}]
    }
  }
}
```

## Limitations & Future Improvements

- **SHAP Computation Speed**: `KernelExplainer` is used as a fallback for non-tree models, which can be computationally expensive on large datasets.
- **Similar Vehicles**: Future phases will integrate the `VehicleKnowledgeEngine` fully to pull actual neighboring valid configurations to enrich the `ValuationReport`.
