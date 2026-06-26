# Intelligent Feature Engineering & Market Intelligence Layer

The Feature Engineering layer (`src/feature_engineering.py`) transforms raw vehicle attributes into deep market-aware features. It relies heavily on the `MarketStatistics` module (`src/market_statistics.py`) which calculates global market baselines across the dataset during initialization to avoid expensive duplicate computations.

## Vehicle Features

### 1. `car_age`
- **Formula**: `CURRENT_YEAR - year`
- **Purpose**: Base time-decay proxy.
- **Expected Impact**: Strong negative correlation with price.

### 2. `km_per_year`
- **Formula**: `km_driven / car_age` (with floor of 0.5 to avoid ZeroDivisionError)
- **Purpose**: Differentiates heavy usage vehicles (taxis/cabs) from low-usage garage queens.
- **Expected Impact**: High values will negatively impact prediction quality.

### 3. `vehicle_segment`
- **Formula**: Inferred via Engine Liter capacity (A-Segment < 1.1L, B-Segment < 1.5L, C-Segment < 2.0L, D/SUV >= 2.0L).
- **Purpose**: Standardizes cars into generic comparable categories for macro-analysis.
- **Expected Impact**: Helps the model understand base price tiers without learning every variant.

### 4. `engine_performance_score`
- **Formula**: `max_power_bhp / engineSize`
- **Purpose**: Calculates specific power output, identifying high-performance or turbocharged engines vs standard naturally aspirated engines.
- **Expected Impact**: Premium feature differentiator.

### 5. `fuel_efficiency_score`
- **Formula**: `mpg / median_mpg_of_fuel_type`
- **Purpose**: Identifies highly efficient vehicles vs gas guzzlers within the same fuel category.
- **Expected Impact**: Indian market is highly sensitive to mileage; expected high positive correlation for mass-market vehicles.

## Ownership Features

### 1. `owner_risk_score`
- **Formula**: Heuristic mapping (First Owner = 1.0, Second = 0.8, Third = 0.6, Fourth+ = 0.4).
- **Purpose**: Quantifies the inherent risk and depreciation associated with multi-hand vehicles.
- **Expected Impact**: Smooths out categorical jumps for linear models.

### 2. `seller_type_score`
- **Formula**: Trust weighting (Trustmark Dealer = 1.0, Dealer = 0.8, Individual = 0.5).
- **Purpose**: Reflects dealer premium (dealers offer warranties/servicing increasing the sale price).
- **Expected Impact**: Models the price bump expected from verified dealers.

## Market Features

### 1. Popularities (`brand_popularity`, `variant_popularity`, `fuel_demand_score`, `transmission_popularity`)
- **Formula**: `count(attribute) / total_vehicles`
- **Purpose**: Maps rarity vs mass-market appeal.
- **Expected Impact**: Popular cars sell faster and retain value better.

### 2. `configuration_scarcity_score`
- **Formula**: `Normalized(1 / variant_popularity)`
- **Purpose**: Identifies rare/niche vehicles.
- **Expected Impact**: Rare variants of luxury cars command premiums; rare variants of budget cars command discounts.

### 3. `market_stability_score`
- **Formula**: `1 - Coefficient of Variation (std / mean) of brand prices`
- **Purpose**: Measures how predictable a brand's pricing is. 
- **Expected Impact**: More stable brands have tighter prediction intervals.

### 4. `market_liquidity_score`
- **Formula**: `(brand_pop * 0.4) + (variant_pop * 0.3) + (fuel_pop * 0.3)`
- **Purpose**: Estimates how easily the vehicle can be sold.
- **Expected Impact**: High liquidity = less depreciation penalty.

## Depreciation Features

### 1. `brand_resale_retention_score`
- **Formula**: Median price of 5-7 year old cars / Median price of 0-2 year old cars (per brand).
- **Purpose**: Empirically measures how well a brand holds value in the Indian market (e.g., Toyota vs Skoda).
- **Expected Impact**: Massively improves long-term valuation accuracy.

### 2. `historical_brand_depreciation`
- **Formula**: `(1 - brand_resale_retention) / 5 years`
- **Purpose**: Base annual depreciation penalty.
- **Expected Impact**: Personalizes the age-penalty per manufacturer.

### 3. `remaining_life_index` (Data-driven)
- **Formula**: `(1.0 - Age_Penalty) * (1.0 - Usage_Penalty * 0.5) * Retention_Score`
- **Purpose**: Advanced, non-linear composite metric describing what percentage of the car's useful life/value remains based on market behaviour, not just fixed lifespans.
- **Expected Impact**: Likely to become the strongest single predictor of price in Phase 5.

## Assumptions & Limitations
- **New Prices**: We assume historical retention based on relative median prices of young vs old cars. This is robust but might break if a brand completely changed its pricing strategy 3 years ago.
- **Regional Data**: The current dataset lacks geographic coordinates, so `regional_demand_score` could not be engineered. If regional data is added, prices in tier 1 vs tier 2 cities should be incorporated.

## Future Recommendations (Phase 5)
- During ML pipeline training (Phase 5), we should use `Feature Importances` (via SHAP or XGBoost) to aggressively prune overlapping engineered features (e.g., dropping raw `year` in favor of `car_age` and `remaining_life_index`).
