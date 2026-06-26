import abc
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from src.utils import logger


class BaseExplanationProvider(abc.ABC):
    @abc.abstractmethod
    def explain(self, model, X: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Explain a single prediction.
        Returns:
            Tuple of (base_value, feature_contributions)
            where feature_contributions is a dictionary of feature_name -> contribution_value
        """
        pass


class ShapExplanationProvider(BaseExplanationProvider):
    def __init__(self):
        try:
            import shap

            self.shap = shap
            self.is_available = True
        except ImportError:
            self.is_available = False
            logger.warning(
                "SHAP is not available. Explanations will fallback to empty."
            )

        self.explainer = None

    def _init_explainer(self, model, X: pd.DataFrame):
        if self.explainer is not None:
            return

        try:
            self.explainer = self.shap.TreeExplainer(model)
        except Exception:
            background = self.shap.sample(X, min(50, len(X)))
            self.explainer = self.shap.KernelExplainer(model.predict, background)

    def explain(self, model, X: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        if not self.is_available:
            return 0.0, {col: 0.0 for col in X.columns}

        self._init_explainer(model, X)

        try:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            base_value = self.explainer.expected_value
            if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            else:
                base_value = float(base_value)

            contributions = {}
            for i, col in enumerate(X.columns):
                contributions[col] = float(shap_values[0][i])

            return base_value, contributions

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return 0.0, {col: 0.0 for col in X.columns}


class ExplanationEngine:
    def __init__(self, provider: BaseExplanationProvider, config: Dict[str, Any]):
        self.provider = provider
        self.translation_dict = config.get("feature_translation_dictionary", {})

    def _translate_feature_name(self, feature: str) -> str:
        return self.translation_dict.get(feature, feature.replace("_", " ").title())

    def _get_direction_text(self, feature: str, contribution: float) -> str:
        translated_name = self._translate_feature_name(feature)
        if contribution > 0:
            return f"{translated_name} increases the market value."
        else:
            return f"{translated_name} reduces the market value."

    def explain_prediction(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate business-friendly explanation of a prediction.
        """
        base_value, contributions = self.provider.explain(model, X)

        # Sort contributions by absolute magnitude
        sorted_contribs = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        positive_factors = []
        negative_factors = []

        for feature, contrib in sorted_contribs:
            if abs(contrib) < 0.01:  # Ignore negligible contributions
                continue

            translated_name = self._translate_feature_name(feature)
            item = {
                "feature": translated_name,
                "raw_feature": feature,
                "contribution": contrib,
                "explanation": self._get_direction_text(feature, contrib),
            }

            if contrib > 0:
                positive_factors.append(item)
            else:
                negative_factors.append(item)

        # Limit to top 3
        positive_factors = positive_factors[:3]
        negative_factors = negative_factors[:3]

        return {
            "base_value": base_value,
            "top_positive_factors": positive_factors,
            "top_negative_factors": negative_factors,
            "raw_contributions": contributions,
        }
