# DRIVEIQ 2.0 — Dataset Engineering Report

> **Phase**: 2 — Dataset Engineering & Data Quality  
> **Author**: DRIVEIQ Engineering  
> **Date**: June 2026  
> **Status**: Complete

---

## 1. Dataset Sources

### Dataset A: CarDekho Vehicle Dataset (v3)

| Property | Value |
|---|---|
| **Source** | [Kaggle: nehalbirla/vehicle-dataset-from-cardekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) |
| **Origin** | Scraped from [cardekho.com](https://www.cardekho.com), India's largest car listing platform |
| **File** | `data/raw/cardekho_v3.csv` |
| **License** | Open Database License (ODbL) v1.0 — permits sharing, modification, and commercial use |
| **Raw Records** | 8,128 |
| **Columns** | 13 |
| **Version** | 4 (updated January 2023) |
| **Downloads** | 228,000+ on Kaggle |

### Dataset B: Multi-City Indian Used Cars

| Property | Value |
|---|---|
| **Source** | [Kaggle: avikasliwal/used-cars-price-prediction](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction) |
| **Origin** | Aggregated Indian used-car listings across 11 cities |
| **File** | `data/raw/kasliwal_train.csv` |
| **License** | Listed as "Other" on Kaggle — publicly available dataset with 228K+ downloads, widely used in academic and ML projects |
| **Raw Records** | 6,019 |
| **Columns** | 14 (including index) |
| **Version** | 2 |

### Licensing Summary

Both datasets are publicly available on Kaggle and are widely used in machine learning
research and education. Dataset A is explicitly licensed under ODbL v1.0. Dataset B
does not specify a restrictive license and is distributed publicly on Kaggle for
download and use.

The raw CSV files are included in `data/raw/` for reproducibility. If redistribution
concerns arise, the download can be automated using `kagglehub`:

```python
import kagglehub
kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")
kagglehub.dataset_download("avikasliwal/used-cars-price-prediction")
```

---

## 2. Dataset Selection Rationale

### Why These Datasets Were Chosen

| Criterion | Dataset A (CarDekho v3) | Dataset B (Kasliwal) |
|---|---|---|
| **Native Indian prices** | ✅ selling_price in INR | ✅ Price in Lakhs (converted to INR) |
| **Indian distance units** | ✅ km_driven in kilometers | ✅ Kilometers_Driven in kilometers |
| **Indian fuel efficiency** | ✅ mileage in kmpl / km/kg | ✅ Mileage in kmpl / km/kg |
| **Indian brands** | ✅ Maruti, Hyundai, Tata, Mahindra | ✅ Same + more luxury brands |
| **Technical specs** | ✅ engine CC, max power, torque, seats | ✅ engine CC, power, seats |
| **Ownership history** | ✅ First/Second/Third/Fourth Owner | ✅ First/Second/Third/Fourth Owner |
| **Geographic coverage** | ❌ No location data | ✅ 11 Indian cities |
| **Seller type** | ✅ Individual/Dealer/Trustmark | ❌ Not available |
| **Data quality** | High — well-structured, widely validated | High — clean, widely used |

### Why These Datasets Complement Each Other

1. **Different source platforms**: Different scraping origins reduce single-source bias
2. **Geographic data**: Dataset B provides city-level location information absent from Dataset A
3. **Seller diversity**: Dataset A provides seller type (Individual vs Dealer) absent from Dataset B
4. **Combined coverage**: Merging produces ~14K records across 37 brands, covering budget through luxury segments

### Datasets Considered but Rejected

| Dataset | Reason for Rejection |
|---|---|
| CarDekho basic (301 rows) | Too small for production ML |
| CAR DETAILS FROM CAR DEKHO.csv (4,340 rows) | Subset of v3 with fewer features |
| Indian IT Cities 2023 | Heavy geographic bias toward IT hubs |
| 140K+ Indian Used Cars (saisumanthvck) | Unverifiable data provenance and quality |
| Quikr scraped datasets | Significant data quality issues (unstructured price fields) |

---

## 3. Merge Strategy

### Feature Mapping Between Datasets

| Unified Column | Dataset A (CarDekho v3) | Dataset B (Kasliwal) | Transformation |
|---|---|---|---|
| `brand` | Extracted from `name` (1st word) | Extracted from `Name` (1st word) | Multi-word brands handled (Land Rover, Ashok Leyland) |
| `model` | Extracted from `name` (words 2–4) | Extracted from `Name` (words 2–4) | Truncated to 3 words to reduce cardinality |
| `year` | `year` (int) | `Year` (int) | Direct mapping |
| `selling_price` | `selling_price` (INR) | `Price × 100,000` (Lakhs → INR) | Dataset B converted from Lakhs |
| `km_driven` | `km_driven` (int) | `Kilometers_Driven` (int) | Direct mapping |
| `fuel_type` | `fuel` | `Fuel_Type` | Normalized to: Petrol, Diesel, CNG, LPG, Electric |
| `transmission` | `transmission` | `Transmission` | Normalized to: Manual, Automatic |
| `owner_type` | `owner` | `Owner_Type` | Normalized to: First Owner, Second Owner, Third Owner, Fourth & Above Owner |
| `mileage_kmpl` | Parsed from `mileage` ("23.4 kmpl") | Parsed from `Mileage` ("23.4 kmpl") | Numeric extraction, handles kmpl and km/kg |
| `engine_cc` | Parsed from `engine` ("1248 CC") | Parsed from `Engine` ("1248 CC") | Numeric extraction |
| `max_power_bhp` | Parsed from `max_power` ("74 bhp") | Parsed from `Power` ("74 bhp") | Numeric extraction, "null" values → None |
| `seats` | `seats` (float) | `Seats` (float) | Direct mapping |
| `seller_type` | `seller_type` | Not available → None | Nullable |
| `location` | Not available → None | `Location` | Nullable |
| `source` | "cardekho_v3" | "kasliwal_multicity" | Data provenance tracking |

### Merge Process

1. Each dataset is loaded by its dedicated loader function
2. Columns are standardized to the unified 15-column schema
3. Categorical values are normalized via lookup maps
4. Datasets are concatenated with `pd.concat(ignore_index=True)`
5. A `source` column tracks data provenance

---

## 4. Cleaning Pipeline

The cleaning pipeline is implemented in `src/data_processing.py::clean_data()` and
executes the following steps in order:

### Step 1: Exact Duplicate Removal

**Method**: `df.drop_duplicates()` across all columns.

**Result**: 1,207 exact duplicates removed.

**Rationale**: Cross-dataset overlap and within-dataset duplicates from scraping.

### Step 2: Missing Critical Fields

**Method**: Drop rows where any of `[brand, model, year, selling_price, km_driven, fuel_type, transmission]` is null. Also remove rows where brand or model is "Unknown" (failed parsing).

**Result**: 0 records removed (all critical fields populated after parsing).

### Step 3: Unrealistic Prices

**Method**: Remove records where `selling_price < ₹20,000` or `selling_price > ₹2,00,00,000` (₹2 Crore).

**Rationale**:
- Below ₹20,000: Likely scrap/junk listings, not genuine resale transactions
- Above ₹2 Crore: Ultra-luxury vehicles outside the training distribution

**Result**: 0 records removed (all prices within bounds after duplicate removal).

### Step 4: Unrealistic Mileage

**Method**: Remove records where `km_driven < 100` or `km_driven > 10,00,000`.

**Rationale**:
- Below 100 km: Likely data entry errors or showroom vehicles
- Above 10 lakh km: Unrealistic for passenger vehicles

**Result**: 4 records removed.

### Step 5: Unrealistic Manufacturing Years

**Method**: Remove records where `year < 1995` or `year > CURRENT_YEAR`.

**Rationale**: Pre-1995 vehicles are extremely rare in Indian resale markets. Future years are data errors.

**Result**: 5 records removed.

### Step 6: Missing Value Imputation

**Method**: Brand-aware median imputation with global median fallback.

For each of `[mileage_kmpl, engine_cc, max_power_bhp, seats]`:
1. Compute the median value per brand
2. Fill missing values with the brand-specific median
3. If the brand has no non-null values, use the global median

**Result**:

| Column | Missing Count | Global Median Fallback |
|---|---|---|
| `mileage_kmpl` | 207 | 18.9 kmpl |
| `engine_cc` | 241 | 1,396 CC |
| `max_power_bhp` | 346 | 86.8 bhp |
| `seats` | 247 | 5.0 |

**Rationale**: Brand-aware imputation is more accurate than global median because
a missing engine CC for a Maruti should be imputed differently than for a BMW.

### Step 7: Brand Name Standardization

**Method**: Normalize brand names via a lookup map.

| Raw Value | Normalized Value |
|---|---|
| "Land" | "Land Rover" |
| "ISUZU" | "Isuzu" |
| "Mercedes-Benz" | "Mercedes" |

### Step 8: Near-Duplicate Removal

**Method**: For records sharing the same (brand, model, year, fuel_type, transmission) key,
remove entries where km_driven is within 2% (or 500 km) AND price is within 5% of a
previously seen record.

**Rationale**: Cross-dataset merging and scraped listings often contain near-duplicate
entries for the same vehicle listed at slightly different times or prices.

**Result**: 293 near-duplicates removed.

### Step 9: Brand-Level Outlier Removal

**Method**: For each brand with ≥10 records, remove prices below the 2nd percentile
and above the 98th percentile of that brand's price distribution.

**Rationale**: Brand-specific outlier removal avoids removing legitimate high-priced
luxury cars when using global percentiles. A ₹50 Lakh Mercedes is normal; a ₹50 Lakh
Maruti is an outlier.

**Result**: 496 brand-level price outliers removed.

---

## 5. Unit Standardization

| Dimension | DRIVEIQ v1 (UK Data) | DRIVEIQ v2 (Indian Data) | Change |
|---|---|---|---|
| **Currency** | GBP × 115 × brand_multiplier → INR | Native INR | No conversion needed |
| **Distance** | Miles (UK data) | Kilometers (native) | No conversion needed |
| **Fuel Efficiency** | Miles per gallon (mpg) | Kilometers per litre (kmpl) | No conversion needed |
| **Engine Size** | Litres (from UK data) | CC (parsed, converted to L for compatibility) | engine_cc / 1000 → engineSize |
| **Power** | Not available in v1 | BHP (parsed from string) | New feature |

All units in v2 are natively Indian. No currency conversion, no distance conversion,
and no fuel efficiency conversion is required.

---

## 6. Final Schema

### Primary Columns (15 unified columns)

| Column | Type | Description | Example | Nullable |
|---|---|---|---|---|
| `brand` | string | Car manufacturer | "Maruti" | No |
| `model` | string | Car model (up to 3 words) | "Swift Dzire VDI" | No |
| `year` | int | Manufacturing year | 2018 | No |
| `selling_price` | int | Resale price in INR | 450000 | No |
| `km_driven` | int | Kilometers driven | 30000 | No |
| `fuel_type` | string | Fuel type | "Diesel" | No |
| `transmission` | string | Transmission type | "Manual" | No |
| `owner_type` | string | Ownership history | "First Owner" | No |
| `mileage_kmpl` | float | Fuel efficiency (kmpl) | 23.4 | No (imputed) |
| `engine_cc` | float | Engine displacement (CC) | 1248.0 | No (imputed) |
| `max_power_bhp` | float | Maximum power (bhp) | 74.0 | No (imputed) |
| `seats` | float | Number of seats | 5.0 | No (imputed) |
| `seller_type` | string | Seller category | "Individual" | Yes |
| `location` | string | Indian city | "Mumbai" | Yes |
| `source` | string | Data provenance | "cardekho_v3" | No |

### Backward-Compatible Aliases (created by `create_features()`)

| Alias Column | Maps To | Purpose |
|---|---|---|
| `price_inr` | `selling_price` | Target variable for existing pipeline |
| `mileage` | `km_driven` | Existing feature name in model |
| `mpg` | `mileage_kmpl` | Existing feature name in model |
| `engineSize` | `engine_cc / 1000` | Existing feature name in model (litres) |
| `fuelType` | `fuel_type` | Existing feature name in model |
| `car_age` | `CURRENT_YEAR - year` | Derived feature |
| `km_per_year` | `km_driven / car_age` | Derived feature |
| `premium_brand_flag` | `1 if brand in PREMIUM_BRANDS else 0` | Derived feature |

---

## 7. Final Dataset Statistics

| Metric | Value |
|---|---|
| **Raw records (pre-cleaning)** | 14,147 |
| **Final records (post-cleaning)** | 12,142 |
| **Records removed** | 2,005 (14.2%) |
| **Exact duplicates removed** | 1,207 |
| **Near-duplicates removed** | 293 |
| **Price outliers removed** | 496 |
| **Invalid records removed** | 9 |
| **Unique brands** | 37 |
| **Unique models** | 1,782 |
| **Price range** | ₹45,000 – ₹1,20,00,000 |
| **Price median** | ₹4,74,000 |
| **Year range** | 1995 – 2020 |
| **Fuel type distribution** | Diesel: 53.7%, Petrol: 45.0%, CNG: 0.9%, LPG: 0.4%, Electric: 0.02% |
| **Transmission distribution** | Manual: 82.6%, Automatic: 17.4% |
| **Sources** | cardekho_v3: 57.5%, kasliwal_multicity: 42.5% |

### Top 10 Brands by Record Count

| Brand | Records | Percentage |
|---|---|---|
| Maruti | 3,147 | 25.9% |
| Hyundai | 2,225 | 18.3% |
| Mahindra | 933 | 7.7% |
| Honda | 911 | 7.5% |
| Tata | 788 | 6.5% |
| Toyota | 721 | 5.9% |
| Ford | 626 | 5.2% |
| Volkswagen | 464 | 3.8% |
| Mercedes | 346 | 2.8% |
| Chevrolet | 320 | 2.6% |

---

## 8. Assumptions

1. **Price target is resale price**: Dataset A uses `selling_price` (actual resale), Dataset B uses `Price` (listed resale in Lakhs). Both represent the actual transaction/listing price, not showroom price.

2. **CNG/LPG mileage equivalence**: CNG mileage is reported as "km/kg" while petrol/diesel uses "kmpl". Both are parsed as numeric fuel efficiency values. For tree-based models, the numeric scale difference is handled by the model.

3. **Brand name is the first word**: We assume the car name always starts with the brand name, with special handling for "Land Rover" and "Ashok Leyland".

4. **Model name truncation to 3 words**: To reduce cardinality from thousands of unique variants to a manageable set, model names are truncated to the first 3 words. This may merge some distinct variants.

5. **Imputation assumes brand-level patterns**: Missing mileage/engine/power values are imputed using the brand's median, assuming vehicles within a brand share similar technical characteristics.

---

## 9. Known Limitations

1. **No post-2020 vehicles**: Both datasets contain listings from pre-2020 scrapes. Newer models (2021+) are not represented.

2. **Geographic bias in Dataset B**: Only 11 cities are represented. Tier-2 and Tier-3 city pricing patterns are not captured.

3. **Missing seller type in Dataset B**: 42.5% of records have no seller type information. This column cannot be used as a mandatory feature.

4. **Missing location in Dataset A**: 57.5% of records have no location. Geographic features cannot be used as mandatory features.

5. **Model cardinality**: 1,782 unique model names remain after truncation. OneHotEncoding produces a very wide feature matrix. Target encoding or embedding approaches would be more efficient (future Phase 4 work).

6. **No service history**: Neither dataset contains service records, accident history, or condition ratings — information that significantly impacts resale value.

7. **No new car MSRP**: Real depreciation calculation requires the original purchase price. Dataset B has a `New_Price` column but with 86% missing values.

8. **Electric vehicles**: Only 2 electric vehicle records exist. The model cannot reliably predict EV prices.

---

## 10. Archived Data

The original DRIVEIQ v1 datasets (8 UK brand CSVs) are preserved in `data/archive/`
with a README documenting their provenance. These files are excluded from the v2
training pipeline but remain available for historical comparison and rollback.

| File | Brand | Records |
|---|---|---|
| `data/archive/audi.csv` | Audi | 10,668 |
| `data/archive/bmw.csv` | BMW | 10,781 |
| `data/archive/ford.csv` | Ford | 17,965 |
| `data/archive/hyundai.csv` | Hyundai | 4,860 |
| `data/archive/mercedes.csv` | Mercedes-Benz | 13,119 |
| `data/archive/skoda.csv` | Skoda | 6,267 |
| `data/archive/toyota.csv` | Toyota | 6,738 |
| `data/archive/vw.csv` | Volkswagen | 15,157 |
