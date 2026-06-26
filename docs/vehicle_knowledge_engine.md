# Vehicle Knowledge Engine

The Vehicle Knowledge Engine acts as the authoritative source of truth for valid vehicle configurations in the DRIVEIQ AI Valuation System.

## Architecture

The engine is built around a normalized **Vehicle Master Table**, generated dynamically from the cleaned Indian market dataset during preprocessing. It establishes a strict hierarchy for vehicle configurations:

**Hierarchy Level:**
1. Brand (e.g., Maruti)
2. Base Model (e.g., Swift)
3. Manufacturing Year (e.g., 2018)
4. Variant (e.g., Dzire VDI)
5. Fuel Type (e.g., Diesel)
6. Transmission (e.g., Manual)

## Features

### 1. Variant-Centric Design
The system treats the **variant** as the central entity for determining exact manufacturer specifications rather than just the generic model.

### 2. Auto-Populated Specifications
Instead of requiring manual input for vehicle specifications (which users often guess incorrectly), the engine automatically infers:
- Engine Size (L)
- Max Power (bhp)
- Mileage (kmpl)
- Seating Capacity

These are calculated using the median value of all matching configurations in the training dataset.

### 3. Validation and Confidence
The engine tracks `confidence_count`, representing the number of historical records matching the exact configuration. This supports future confidence estimation capabilities.

### 4. Backward Compatibility
The `get_full_model_string()` method seamlessly reconstructs the original model string (e.g., "Swift Dzire VDI") required by the Phase 2 machine learning models, ensuring zero breaking changes to the underlying prediction pipeline.

## API Endpoints

The engine is exposed via REST API to ensure mobile apps, external clients, and microservices all utilize the exact same knowledge graph:

- `GET /knowledge/brands`: List all valid brands.
- `GET /knowledge/models?brand={brand}`: List models for a brand.
- `GET /knowledge/years?brand={brand}&model={model}`: List valid years.
- `GET /knowledge/variants?brand={brand}&model={model}&year={year}`: List valid variants.
- `GET /knowledge/specs?...`: Retrieve specifications for a fully-qualified configuration.

## Extensibility
The engine is designed to easily accommodate new manufacturer specifications (e.g., Battery Capacity, NCAP Safety Ratings) by simply adding them to the aggregation logic in `_build_master_table`.
