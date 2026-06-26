# Decision Intelligence Layer

The Automotive Decision Intelligence Platform transforms raw vehicle valuations into practical buying recommendations and negotiation aids.

## Architecture

The system is modularized under `src/decision_intelligence/`:

1. **Recommendation Engine**: Generates the final deal score and qualitative recommendation (`Excellent Buy`, `Good Buy`, `Negotiate`, `Wait`, `Avoid`).
2. **Negotiation Engine**: If an asking price is available, computes the optimal starting offer, maximum recommended offer, and assesses negotiation difficulty.
3. **Forecast Engine**: Generates a 5-year timeline estimating retained value percentage and raw expected price.
4. **Ownership Engine**: Approximates total 5-year running costs including insurance, maintenance, fuel, and registration.
5. **Buyer Insights Engine**: Triggers specific, human-readable insights (e.g., `Highway Friendly`, `Strong Resale Value`) based on vehicle characteristics.
6. **Alternatives Engine**: Determines appropriate alternative models based on budget constraints and segment.

## Deal Score Calculation

The `Deal Score` (0-100) quantifies the attractiveness of the vehicle purchase:
* **Base Score**: 50
* **Price Fairness**: If asking price < 95% of estimated value (+25 points), > 115% (-20 points).
* **Confidence Bonus**: Slight bump for high predictive confidence.
* **Risk Adjustments**: Major penalties for High Maintenance/Ownership Risk.

Scores above 85 are classified as an `Excellent Buy`. Scores below 35 prompt an `Avoid` recommendation.

## Ownership Assumptions
* Insurance is estimated at 3% of the car's current value annually.
* Base maintenance is 15,000 INR/year, scaling linearly with engine size and vehicle age.
* Fuel cost assumes a fixed driving distance (12,000 km/year) at 100 INR/L.

## Integration
This layer acts directly between the ML Prediction Pipeline and the UI Dashboard (`src/ui/dashboard.py`). It receives the raw `valuation_report` payload and enhances it with an attached `decision_report`.
