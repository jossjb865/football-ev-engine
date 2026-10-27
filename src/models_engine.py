"""Motor de modelos probabilisticos y calculo de valor esperado.

Mejoras respecto a la version original:
- Carga tambien el Poisson bivariado entrenado (``ucl_bivariate_poisson.joblib``)
  y usa su covarianza ajustada por maxima verosimilitud en lugar del valor
  hardcodeado de 0.18.
- Probabilidades acotadas a [0.01, 0.99] para evitar sobreconfianza extrema.
- Criterio de Kelly para dimensionar la stake recomendada.
- ``models_dir`` configurable por variable de entorno MODELS_DIR.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import joblib
import numpy as np
from scipy.stats import poisson

try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:  # el scanner puede correr sin modelos entrenados
    CatBoostRegressor = None
    _HAS_CATBOOST = False

# Limites de probabilidad: protegen contra calibraciones degeneradas.
PROB_MIN, PROB_MAX = 0.01, 0.99


class ModelsEngine:
    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.getenv("MODELS_DIR", "models")
        self.corners_model = None
        self.sot_model = None
        self.btts_model = None
        self.bivariate = None
        self._load_models()

    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        """Carga modelos serializados si existen (fallos no fatales)."""
        paths = {
            "corners_model": os.path.join(self.models_dir, "ucl_home_corners_poisson.cbm"),
            "sot_model": os.path.join(self.models_dir, "ucl_home_sot_poisson.cbm"),
            "btts_model": os.path.join(self.models_dir, "ucl_btts_calibrated.joblib"),
            "bivariate": os.path.join(self.models_dir, "ucl_bivariate_poisson.joblib"),
        }
        if _HAS_CATBOOST and os.path.exists(paths["corners_model"]):
            self.corners_model = CatBoostRegressor()
            self.corners_model.load_model(paths["corners_model"])
        if _HAS_CATBOOST and os.path.exists(paths["sot_model"]):
            self.sot_model = CatBoostRegressor()
            self.sot_model.load_model(paths["sot_model"])
        if os.path.exists(paths["btts_model"]):
            self.btts_model = joblib.load(paths["btts_model"])
        if os.path.exists(paths["bivariate"]):
            self.bivariate = joblib.load(paths["bivariate"])

    # ------------------------------------------------------------------
    # Poisson
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_poisson_over(lambda_val: float, line: float) -> float:
        """P(X > line) usando la funcion de supervivencia de Poisson."""
        if lambda_val <= 0:
            return 0.0
        return float(poisson.sf(math.floor(line), lambda_val))

    def calculate_bivariate_btts(self, lambda_home: float, lambda_away: float,
                                 covariance: float = 0.18) -> float:
        """P(BTTS Si) = 1 - e^{-(l1+l3)} - e^{-(l2+l3)} + e^{-(l1+l2+l3)}.

        Si existe el modelo bivariado entrenado, usa su covarianza ajustada.
        """
        if self.bivariate is not None:
            covariance = float(getattr(self.bivariate, "l3_cov", covariance))
        l1 = max(0.05, lambda_home - covariance)
        l2 = max(0.05, lambda_away - covariance)
        l3 = max(0.01, covariance)

        p_h_zero = math.exp(-(l1 + l3))
        p_a_zero = math.exp(-(l2 + l3))
        p_both_zero = math.exp(-(l1 + l2 + l3))
        prob = 1.0 - p_h_zero - p_a_zero + p_both_zero
        return min(max(prob, PROB_MIN), PROB_MAX)

    # ------------------------------------------------------------------
    # Predicciones
    # ------------------------------------------------------------------
    def predict_corners_lambda(self, feature_vector: np.ndarray,
                               fallback_lambda: float) -> float:
        if self.corners_model is not None:
            return max(0.05, float(self.corners_model.predict(feature_vector)[0]))
        return max(0.05, fallback_lambda)

    def predict_sot_lambda(self, feature_vector: np.ndarray,
                           fallback_lambda: float) -> float:
        if self.sot_model is not None:
            return max(0.05, float(self.sot_model.predict(feature_vector)[0]))
        return max(0.05, fallback_lambda)

    def predict_btts_probability(self, feature_vector: np.ndarray,
                                 lambda_h: float, lambda_a: float) -> float:
        if self.btts_model is not None:
            prob = float(self.btts_model.predict_proba(feature_vector)[0][1])
            return min(max(prob, PROB_MIN), PROB_MAX)
        return self.calculate_bivariate_btts(lambda_h, lambda_a)

    # ------------------------------------------------------------------
    # Valor esperado y Kelly
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_ev(prob: float, odds: float) -> float:
        """EV = p * cuota - 1. Positivo implica apuesta con valor."""
        if odds <= 1.0:
            return -1.0
        return prob * odds - 1.0

    @staticmethod
    def kelly_stake(prob: float, odds: float, fraction: float = 0.25) -> float:
        """Fraccion de banca recomendada (Kelly fraccional, 0 si no hay valor)."""
        ev = ModelsEngine.calculate_ev(prob, odds)
        if ev <= 0:
            return 0.0
        full_kelly = (prob * odds - 1.0) / (odds - 1.0)
        return max(0.0, full_kelly * fraction)
