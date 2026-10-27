"""Tests unitarios del motor probabilistico y de features (sin red/API)."""
import math

import numpy as np

from src.models_engine import ModelsEngine
from src.feature_builder import FeatureBuilder
from src.odds_consolidator import consolidate
from src.config import ScannerConfig


def test_poisson_over_valor_conocido():
    # P(X > 2.5) con lambda=3 -> P(X>=3) = 1 - e^-3 * (1 + 3 + 9/2)
    esperado = 1 - math.exp(-3) * 8.5
    assert abs(ModelsEngine.calculate_poisson_over(3.0, 2.5) - esperado) < 1e-9


def test_poisson_over_lambda_cero():
    assert ModelsEngine.calculate_poisson_over(0.0, 2.5) == 0.0


def test_ev():
    assert abs(ModelsEngine.calculate_ev(0.5, 2.2) - 0.10) < 1e-9
    assert ModelsEngine.calculate_ev(0.5, 1.0) == -1.0  # cuota invalida


def test_btts_coincide_con_independencia_cuando_cov_cero():
    p = ModelsEngine().calculate_bivariate_btts(1.5, 1.2, covariance=0.0)
    esperado = (1 - math.exp(-1.5)) * (1 - math.exp(-1.2))
    assert abs(p - esperado) < 1e-6


def test_btts_acotado():
    p = ModelsEngine().calculate_bivariate_btts(5.0, 5.0, covariance=0.18)
    assert 0.01 <= p <= 0.99


def test_kelly():
    assert ModelsEngine.kelly_stake(0.5, 2.0) > 0
    assert ModelsEngine.kelly_stake(0.3, 2.0) == 0.0  # sin valor -> 0
    # Kelly completo: (p*o - 1)/(o - 1) = (0.6*2 - 1)/1 = 0.2; quarter = 0.05
    assert abs(ModelsEngine.kelly_stake(0.6, 2.0, fraction=1.0) - 0.2) < 1e-9


def test_feature_fallback_sin_historial():
    fb = FeatureBuilder().extract_metrics([], "t1")
    assert fb["gf"] == 1.2 and fb["corners"] == 4.5 and fb["poss"] == 50.0


def test_feature_ponderado_por_recencia():
    hist = [
        {"id": "m1", "home_team": {"id": "t1"}, "score": {"home": 0, "away": 0}},
        {"id": "m2", "home_team": {"id": "t1"}, "score": {"home": 2, "away": 1}},
    ]
    m = FeatureBuilder(half_life=1.0, stats_provider=lambda mid: {}).extract_metrics(hist, "t1")
    assert m["gf"] > 1.0  # el partido reciente (2 goles) pesa mas que el antiguo (0)


def test_consolidate_tolerante_y_maximo():
    cfg = ScannerConfig()
    bms = [
        {"bookmaker": "A", "markets": {
            "total_goals": {"2.5": {"over": {"last_seen": "1.90"}},
                            "2.25": {"over": {"last_seen": "2.00"}},   # asiatico, ignorar
                            "3.0": {"over": {"last_seen": "2.10"}}},   # entero, ignorar
            "btts": {"yes": {"last_seen": "1.85"}}}},
        {"bookmaker": "B", "markets": {
            "total_goals": {"2.5": {"over": {"last_seen": "2.05"}}},
            "btts": {"yes": {"last_seen": "bad-data"}}}},               # malformado, ignorar
    ]
    best = consolidate(bms, cfg)
    assert best["total_goals"][2.5]["odd"] == 2.05
    assert best["total_goals"][2.5]["bookmaker"] == "B"
    assert len(best["total_goals"]) == 1  # solo linea pura .5
    assert best["btts_yes"]["odd"] == 1.85
