import os
import math
from scipy.stats import poisson
import numpy as np
import joblib
from catboost import CatBoostRegressor

class ModelsEngine:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.corners_model = None
        self.sot_model = None
        self.btts_model = None
        self._load_models()

    def _load_models(self):
        """Carga modelos serializados si existen en la carpeta models/"""
        corners_path = os.path.join(self.models_dir, "ucl_home_corners_poisson.cbm")
        sot_path = os.path.join(self.models_dir, "ucl_home_sot_poisson.cbm")
        btts_path = os.path.join(self.models_dir, "ucl_btts_calibrated.joblib")

        if os.path.exists(corners_path):
            self.corners_model = CatBoostRegressor()
            self.corners_model.load_model(corners_path)

        if os.path.exists(sot_path):
            self.sot_model = CatBoostRegressor()
            self.sot_model.load_model(sot_path)

        if os.path.exists(btts_path):
            self.btts_model = joblib.load(btts_path)

    @staticmethod
    def calculate_poisson_over(lambda_val: float, line: float) -> float:
        """P(X > line) usando la función de supervivencia de Poisson"""
        if lambda_val <= 0:
            return 0.0
        return float(poisson.sf(math.floor(line), lambda_val))

    @staticmethod
    def calculate_bivariate_btts(lambda_home: float, lambda_away: float, covariance: float = 0.18) -> float:
        """P(BTTS Sí) = 1 - e^{-(l1+l3)} - e^{-(l2+l3)} + e^{-(l1+l2+l3)}"""
        l1 = max(0.05, lambda_home - covariance)
        l2 = max(0.05, lambda_away - covariance)
        l3 = max(0.01, covariance)

        p_h_zero = math.exp(-(l1 + l3))
        p_a_zero = math.exp(-(l2 + l3))
        p_both_zero = math.exp(-(l1 + l2 + l3))
        return 1.0 - p_h_zero - p_a_zero + p_both_zero

    def predict_corners_lambda(self, feature_vector: np.ndarray, fallback_lambda: float) -> float:
        if self.corners_model:
            return float(self.corners_model.predict(feature_vector)[0])
        return fallback_lambda

    def predict_sot_lambda(self, feature_vector: np.ndarray, fallback_lambda: float) -> float:
        if self.sot_model:
            return float(self.sot_model.predict(feature_vector)[0])
        return fallback_lambda

    def predict_btts_probability(self, feature_vector: np.ndarray, lambda_h: float, lambda_a: float) -> float:
        if self.btts_model:
            return float(self.btts_model.predict_proba(feature_vector)[0][1])
        return self.calculate_bivariate_btts(lambda_h, lambda_a)

    @staticmethod
    def calculate_ev(prob: float, odds: float) -> float:
        return (prob * odds) - 1.0
