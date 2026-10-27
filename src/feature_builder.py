"""Construccion de variables de entrada para los modelos.

Mejoras respecto a la version original:
- Cache de estadisticas por ``match_id``: elimina la explosion de llamadas
  API (antes 1 llamada por partido de historial por equipo, sin deduplicar).
- Ponderado por recencia (decaimiento exponencial) en lugar de media plana:
  los partidos recientes reflejan mejor el estado de forma actual.
- Medias vectorizadas con NumPy y fallbacks centralizados en ``config``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config import FALLBACK_METRICS

# Vector de caracteristicas de 9 dimensiones (debe coincidir con el
# orden usado en el entrenamiento de CatBoost).
FEATURE_NAMES = [
    "home_gf", "home_ga", "home_corners", "home_sot", "home_poss",
    "away_ga", "away_corners", "away_sot", "away_poss",
]


class FeatureBuilder:
    def __init__(
        self,
        half_life: float = 4.0,
        stats_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        """
        :param half_life: media vida en partidos para el ponderado por recencia.
                          Usar ``None`` o 0 para media plana (comportamiento original).
        :param stats_provider: funcion opcional ``match_id -> stats`` para
                               inyectar un proveedor de estadisticas custom.
        """
        self.half_life = half_life
        self._stats_provider = stats_provider
        self._stats_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def _get_stats(self, client, match_id: str) -> Dict[str, Any]:
        """Estadisticas con cache en memoria por partido (deduplicacion)."""
        if match_id in self._stats_cache:
            return self._stats_cache[match_id]
        if self._stats_provider is not None:
            stats = self._stats_provider(match_id) or {}
        else:
            stats = client.get_match_stats(match_id) if client else {}
        self._stats_cache[match_id] = stats
        return stats

    # ------------------------------------------------------------------
    def extract_metrics(
        self,
        history: List[Dict[str, Any]],
        target_team_id: str,
        client=None,
    ) -> Dict[str, float]:
        """Promedios ponderados por recencia de goles, corners y remates.

        Mantiene la firma original ``(history, target_team_id, client)`` para
        compatibilidad con el codigo existente.
        """
        if not history:
            return dict(FALLBACK_METRICS)

        n = len(history)
        if self.half_life and self.half_life > 0:
            # Peso mayor a los partidos mas recientes (ultimo = peso 1).
            ages = np.arange(n - 1, -1, -1, dtype=float)
            weights = np.power(0.5, ages / self.half_life)
        else:
            weights = np.ones(n, dtype=float)

        gf, ga, corners, sot, poss = [], [], [], [], []
        for m in history:
            match_id = m.get("id")
            is_home = m.get("home_team", {}).get("id") == target_team_id

            score = m.get("score", {})
            h_score = score.get("home", 0) or 0
            a_score = score.get("away", 0) or 0
            gf.append(h_score if is_home else a_score)
            ga.append(a_score if is_home else h_score)

            stats = self._get_stats(client, match_id)
            if stats:
                side = "home" if is_home else "away"
                t_stats = stats.get(side, {})
                corners.append(t_stats.get("corners", 0) or 0)
                sot.append(t_stats.get("shots_on_target", 0) or 0)
                poss.append(t_stats.get("possession", 50) or 50)

        def wmean(values: List[float], fallback: float) -> float:
            if not values:
                return fallback
            return float(np.average(
                np.asarray(values, dtype=float), weights=weights[: len(values)]
            ))

        return {
            "gf": wmean(gf, FALLBACK_METRICS["gf"]),
            "ga": wmean(ga, FALLBACK_METRICS["ga"]),
            "corners": wmean(corners, FALLBACK_METRICS["corners"]),
            "sot": wmean(sot, FALLBACK_METRICS["sot"]),
            "poss": wmean(poss, FALLBACK_METRICS["poss"]),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def build_feature_vector(h_stats: Dict[str, float],
                             a_stats: Dict[str, float]) -> np.ndarray:
        """Vector de 9 dimensiones en el orden esperado por los modelos."""
        return np.array([[
            h_stats["gf"], h_stats["ga"], h_stats["corners"],
            h_stats["sot"], h_stats["poss"],
            a_stats["ga"], a_stats["corners"], a_stats["sot"], a_stats["poss"],
        ]], dtype=float)
