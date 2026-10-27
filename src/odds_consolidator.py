"""Consolidacion de la mejor cuota por mercado entre todas las bookmakers.

Logica pura y testeable: recibe la lista de bookmakers y devuelve, por
mercado y linea, la cuota mas alta junto a la casa que la ofrece.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from .config import ScannerConfig


def _safe_float(value: Any) -> Optional[float]:
    """Conversion tolerante: nunca lanza excepcion (antes tumbaria el scan)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def consolidate(bookmakers: List[Dict[str, Any]],
                cfg: ScannerConfig) -> Dict[str, Any]:
    """Devuelve la mejor cuota (odd + bookmaker) por mercado/linea.

    Reglas de negocio preservadas:
    - Solo lineas puras .5 (se excluyen asiaticos .25/.75 y enteros).
    - Solo cuotas dentro del rango [cfg.min_odd, cfg.max_odd].
    - Se conserva el maximo entre todas las casas.
    """
    best: Dict[str, Any] = {
        "btts_yes": {"odd": 0.0, "bookmaker": ""},
        "total_goals": defaultdict(lambda: {"odd": 0.0, "bookmaker": ""}),
        "corners": defaultdict(lambda: {"odd": 0.0, "bookmaker": ""}),
        "shots_on_target": defaultdict(lambda: {"odd": 0.0, "bookmaker": ""}),
    }

    for bm in bookmakers:
        bm_name = bm.get("bookmaker", "Desconocido")
        markets = bm.get("markets", {})

        # 1. BTTS: Si
        yes = (markets.get("btts", {}).get("yes", {}) or {}).get("last_seen")
        odd = _safe_float(yes)
        if odd and odd > best["btts_yes"]["odd"]:
            best["btts_yes"] = {"odd": odd, "bookmaker": bm_name}

        # 2-4. Mercados de Over por linea
        for key, mkt in (("total_goals", "total_goals"),
                         ("corners", "corners"),
                         ("shots_on_target", "shots_on_target")):
            for line_str, outcomes in (markets.get(mkt, {}) or {}).items():
                line = _safe_float(line_str)
                if line is None or line % 1.0 != 0.5:
                    continue
                over = _safe_float((outcomes.get("over", {}) or {}).get("last_seen"))
                if (over and cfg.min_odd <= over <= cfg.max_odd
                        and over > best[key][line]["odd"]):
                    best[key][line] = {"odd": over, "bookmaker": bm_name}

    return best
