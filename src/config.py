"""Configuracion central del motor +EV.

Unifica los valores magicos que antes estaban dispersos por el codigo y
permite sobreescribirlos por variables de entorno.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


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
