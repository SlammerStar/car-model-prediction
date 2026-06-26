import pandas as pd
from typing import List, Dict, Any, Optional


class VehicleKnowledgeEngine:
    """
    Vehicle Knowledge Base
    Acts as the authoritative source of truth for valid vehicle configurations.
    Maintains a normalized internal Knowledge Base to enforce valid
    combinations of Brand -> Base Model -> Year -> Variant -> Fuel -> Transmission.
    """

    BRAND_KNOWLEDGE = {
        "Maruti Suzuki": {
            "reliability": 0.85,
            "liquidity": 0.95,
            "maintenance_cost": "low",
            "premium": False,
        },
        "Hyundai": {
            "reliability": 0.80,
            "liquidity": 0.85,
            "maintenance_cost": "low",
            "premium": False,
        },
        "Honda": {
            "reliability": 0.90,
            "liquidity": 0.80,
            "maintenance_cost": "low",
            "premium": False,
        },
        "Toyota": {
            "reliability": 0.95,
            "liquidity": 0.90,
            "maintenance_cost": "medium",
            "premium": False,
        },
        "Tata": {
            "reliability": 0.70,
            "liquidity": 0.75,
            "maintenance_cost": "low",
            "premium": False,
        },
        "Mahindra": {
            "reliability": 0.75,
            "liquidity": 0.80,
            "maintenance_cost": "medium",
            "premium": False,
        },
        "Ford": {
            "reliability": 0.75,
            "liquidity": 0.60,
            "maintenance_cost": "medium",
            "premium": False,
        },
        "Volkswagen": {
            "reliability": 0.65,
            "liquidity": 0.65,
            "maintenance_cost": "high",
            "premium": False,
        },
        "Skoda": {
            "reliability": 0.60,
            "liquidity": 0.60,
            "maintenance_cost": "high",
            "premium": False,
        },
        "BMW": {
            "reliability": 0.60,
            "liquidity": 0.40,
            "maintenance_cost": "very high",
            "premium": True,
        },
        "Audi": {
            "reliability": 0.60,
            "liquidity": 0.40,
            "maintenance_cost": "very high",
            "premium": True,
        },
        "Mercedes-Benz": {
            "reliability": 0.65,
            "liquidity": 0.45,
            "maintenance_cost": "very high",
            "premium": True,
        },
        "Volvo": {
            "reliability": 0.70,
            "liquidity": 0.30,
            "maintenance_cost": "very high",
            "premium": True,
        },
        "Jaguar": {
            "reliability": 0.50,
            "liquidity": 0.20,
            "maintenance_cost": "very high",
            "premium": True,
        },
        "Land Rover": {
            "reliability": 0.40,
            "liquidity": 0.25,
            "maintenance_cost": "very high",
            "premium": True,
        },
        "Porsche": {
            "reliability": 0.70,
            "liquidity": 0.15,
            "maintenance_cost": "very high",
            "premium": True,
        },
        "Lexus": {
            "reliability": 0.90,
            "liquidity": 0.20,
            "maintenance_cost": "high",
            "premium": True,
        },
        "Mini": {
            "reliability": 0.65,
            "liquidity": 0.25,
            "maintenance_cost": "high",
            "premium": True,
        },
        "Jeep": {
            "reliability": 0.60,
            "liquidity": 0.40,
            "maintenance_cost": "high",
            "premium": False,
        },
        "MG": {
            "reliability": 0.65,
            "liquidity": 0.55,
            "maintenance_cost": "medium",
            "premium": False,
        },
        "Kia": {
            "reliability": 0.80,
            "liquidity": 0.80,
            "maintenance_cost": "medium",
            "premium": False,
        },
        "Renault": {
            "reliability": 0.65,
            "liquidity": 0.65,
            "maintenance_cost": "medium",
            "premium": False,
        },
        "Nissan": {
            "reliability": 0.70,
            "liquidity": 0.50,
            "maintenance_cost": "medium",
            "premium": False,
        },
        "Datsun": {
            "reliability": 0.60,
            "liquidity": 0.40,
            "maintenance_cost": "low",
            "premium": False,
        },
        "Chevrolet": {
            "reliability": 0.50,
            "liquidity": 0.30,
            "maintenance_cost": "high",
            "premium": False,
        },
        "Fiat": {
            "reliability": 0.40,
            "liquidity": 0.20,
            "maintenance_cost": "high",
            "premium": False,
        },
        "Mitsubishi": {
            "reliability": 0.60,
            "liquidity": 0.20,
            "maintenance_cost": "high",
            "premium": False,
        },
        "Ssangyong": {
            "reliability": 0.50,
            "liquidity": 0.10,
            "maintenance_cost": "high",
            "premium": False,
        },
        "Isuzu": {
            "reliability": 0.80,
            "liquidity": 0.20,
            "maintenance_cost": "medium",
            "premium": False,
        },
        "Force": {
            "reliability": 0.70,
            "liquidity": 0.30,
            "maintenance_cost": "medium",
            "premium": False,
        },
    }

    def __init__(self, df: pd.DataFrame):
        self._build_knowledge_base(df)

    def get_brand_knowledge(self, brand: str) -> Dict[str, Any]:
        """Returns market knowledge about a brand."""
        return self.BRAND_KNOWLEDGE.get(
            brand,
            {
                "reliability": 0.65,
                "liquidity": 0.50,
                "maintenance_cost": "medium",
                "premium": False,
            },
        )

    def _build_knowledge_base(self, df: pd.DataFrame):
        """
        Builds a normalized knowledge base from the preprocessed dataframe.
        Groups by configuration hierarchy and aggregates specifications.
        """
        # We need these columns to exist in the dataframe
        required_cols = [
            "brand",
            "base_model",
            "variant",
            "year",
            "fuelType",
            "transmission",
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in dataframe: {col}")

        # Group by configuration hierarchy to establish valid combinations
        # We also calculate the median of specifications and count occurrences for confidence
        self.kb = (
            df.groupby(required_cols, dropna=False)
            .agg(
                engineSize=("engineSize", "median"),
                max_power_bhp=("max_power_bhp", "median"),
                mileage=("mpg", "median"),
                seats=("seats", "median"),
                confidence_count=("brand", "count"),
            )
            .reset_index()
        )

        # Sort for deterministic ordering in UI
        self.kb = self.kb.sort_values(by=required_cols)

    def get_brands(self) -> List[str]:
        """Get list of all valid brands."""
        return sorted(self.kb["brand"].dropna().unique().tolist())

    def get_models(self, brand: str) -> List[str]:
        """Get valid base models for a brand."""
        subset = self.kb[self.kb["brand"] == brand]
        return sorted(subset["base_model"].dropna().unique().tolist())

    def get_years(self, brand: str, base_model: str) -> List[int]:
        """Get valid manufacturing years for a brand + base_model combination."""
        subset = self.kb[
            (self.kb["brand"] == brand) & (self.kb["base_model"] == base_model)
        ]
        # Return years sorted descending (newest first)
        years = subset["year"].dropna().unique().tolist()
        return sorted([int(y) for y in years], reverse=True)

    def get_variants(self, brand: str, base_model: str, year: int) -> List[str]:
        """Get valid variants for a brand + base_model + year combination."""
        subset = self.kb[
            (self.kb["brand"] == brand)
            & (self.kb["base_model"] == base_model)
            & (self.kb["year"] == year)
        ]
        return sorted(subset["variant"].dropna().unique().tolist())

    def get_fuel_types(
        self, brand: str, base_model: str, year: int, variant: str
    ) -> List[str]:
        """Get valid fuel types for the selected configuration."""
        subset = self.kb[
            (self.kb["brand"] == brand)
            & (self.kb["base_model"] == base_model)
            & (self.kb["year"] == year)
            & (self.kb["variant"] == variant)
        ]
        return sorted(subset["fuelType"].dropna().unique().tolist())

    def get_transmissions(
        self, brand: str, base_model: str, year: int, variant: str, fuel: str
    ) -> List[str]:
        """Get valid transmissions for the selected configuration."""
        subset = self.kb[
            (self.kb["brand"] == brand)
            & (self.kb["base_model"] == base_model)
            & (self.kb["year"] == year)
            & (self.kb["variant"] == variant)
            & (self.kb["fuelType"] == fuel)
        ]
        return sorted(subset["transmission"].dropna().unique().tolist())

    def get_specs(
        self,
        brand: str,
        base_model: str,
        year: int,
        variant: str,
        fuel: str,
        transmission: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get automatically populated specifications for a completed valid configuration.
        """
        subset = self.kb[
            (self.kb["brand"] == brand)
            & (self.kb["base_model"] == base_model)
            & (self.kb["year"] == year)
            & (self.kb["variant"] == variant)
            & (self.kb["fuelType"] == fuel)
            & (self.kb["transmission"] == transmission)
        ]

        if len(subset) == 0:
            return None

        row = subset.iloc[0]
        conf_count = int(row["confidence_count"])
        if conf_count >= 10:
            conf_level = "High Confidence"
        elif conf_count >= 3:
            conf_level = "Medium Confidence"
        else:
            conf_level = "Low Confidence"

        return {
            "engineSize": (
                round(row["engineSize"], 1) if pd.notna(row["engineSize"]) else 0.0
            ),
            "max_power_bhp": (
                round(row["max_power_bhp"], 1)
                if pd.notna(row["max_power_bhp"])
                else 0.0
            ),
            "mileage": round(row["mileage"], 1) if pd.notna(row["mileage"]) else 0.0,
            "seats": int(row["seats"]) if pd.notna(row["seats"]) else 5,
            "confidence_count": conf_count,
            "confidence_level": conf_level,
        }

    def get_variant_label(
        self, brand: str, base_model: str, year: int, variant: str
    ) -> str:
        """
        Returns an enriched variant label including fuel, transmission, and engine size.
        E.g., "Dzire VDI (1.2L Diesel Manual)"
        """
        subset = self.kb[
            (self.kb["brand"] == brand)
            & (self.kb["base_model"] == base_model)
            & (self.kb["year"] == year)
            & (self.kb["variant"] == variant)
        ]
        if len(subset) == 0:
            return variant

        # Taking the first/most common specs for this variant to build the label
        row = subset.iloc[0]
        engine = (
            f"{round(row['engineSize'], 1)}L " if pd.notna(row["engineSize"]) else ""
        )
        fuel = str(row["fuelType"])
        trans = str(row["transmission"])

        return f"{variant} ({engine}{fuel} {trans})"

    def get_similar_configurations(
        self, brand: str, base_model: str, year: int, variant: str
    ) -> List[Dict[str, Any]]:
        """
        Returns similar valid configurations for the selected vehicle (e.g. other variants).
        Useful for recommendations.
        """
        subset = self.kb[
            (self.kb["brand"] == brand)
            & (self.kb["base_model"] == base_model)
            & (self.kb["year"] == year)
            & (self.kb["variant"] != variant)
        ]

        # Return top 5 most common alternative variants
        subset = subset.sort_values(by="confidence_count", ascending=False).head(5)

        similar = []
        for _, row in subset.iterrows():
            similar.append(
                {
                    "variant": row["variant"],
                    "fuelType": row["fuelType"],
                    "transmission": row["transmission"],
                    "engineSize": (
                        round(row["engineSize"], 1)
                        if pd.notna(row["engineSize"])
                        else 0.0
                    ),
                    "max_power_bhp": (
                        round(row["max_power_bhp"], 1)
                        if pd.notna(row["max_power_bhp"])
                        else 0.0
                    ),
                }
            )
        return similar

    def get_full_model_string(self, base_model: str, variant: str) -> str:
        """
        Reconstructs the full ML model string expected by the prediction pipeline.
        This ensures backward compatibility with the existing ML models.
        """
        if variant == "Standard":
            return base_model
        return f"{base_model} {variant}".strip()
