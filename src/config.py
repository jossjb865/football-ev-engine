"""Configuracion central del motor +EV.

Unifica los valores magicos que antes estaban dispersos por el codigo y
permite sobreescribirlos por variables de entorno.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


# Valores por defecto cuando un equipo no tiene historial suficiente.
FALLBACK_METRICS = {"gf": 1.2, "ga": 1.2, "corners": 4.5, "sot": 4.0, "poss": 50.0}


@dataclass(frozen=True)
class ScannerConfig:
    """Parametros de negocio del scanner (inmutables durante una corrida)."""

    min_odd: float = _env_float("MIN_ODD", 1.40)
    max_odd: float = _env_float("MAX_ODD", 3.80)
    min_ev: float = _env_float("MIN_EV", 5.0) / 100.0
    history_limit: int = _env_int("HISTORY_LIMIT", 8)
    # Media vida (en partidos) para el ponderado por recencia de las metricas.
    recency_half_life: float = _env_float("RECENCY_HALF_LIFE", 4.0)
    # Covarianza de fallback para el Poisson bivariado si no hay modelo entrenado.
    btts_covariance: float = _env_float("BTTS_COVARIANCE", 0.18)
    # Fraccion Kelly a recomendar (0.25 = Quarter Kelly, conservador).
    kelly_fraction: float = _env_float("KELLY_FRACTION", 0.25)

    def __post_init__(self):
        if self.min_odd >= self.max_odd:
            raise ValueError("MIN_ODD debe ser menor que MAX_ODD")
        if self.min_ev < 0:
            raise ValueError("MIN_EV no puede ser negativo")


@dataclass(frozen=True)
class BacktestConfig:
    """Parametros del backtester post-partido.

    Controla donde se persisten predicciones/resultados, cuantas muestras
    se acumulan antes de disparar reentrenamiento y el umbral de Brier
    que indica degradacion de calibracion.
    """

    # Directorio donde se guardan predictions.jsonl / outcomes.jsonl / metrics.json
    data_dir: Path = _env_path("BACKTEST_DATA_DIR", "data/backtest")
    # Minimo de partidos resueltos para considerar reentrenamiento.
    retrain_threshold: int = _env_int("BACKTEST_RETRAIN_THRESHOLD", 50)
    # Si el Brier score supera este valor, se fuerza reentrenamiento
    # aun antes de alcanzar retrain_threshold.
    brier_degradation_threshold: float = _env_float("BACKTEST_BRIER_THRESHOLD", 0.25)
    # Minimo de muestras para evaluar el Brier de forma fiable.
    brier_min_samples: int = _env_int("BACKTEST_BRIER_MIN_SAMPLES", 20)

    def __post_init__(self):
        if self.retrain_threshold < 1:
            raise ValueError("BACKTEST_RETRAIN_THRESHOLD debe ser >= 1")
        if not 0.0 <= self.brier_degradation_threshold <= 1.0:
            raise ValueError("BACKTEST_BRIER_THRESHOLD debe estar en [0, 1]")   
