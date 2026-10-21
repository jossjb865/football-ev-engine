import numpy as np
from typing import Dict, Any, List

class FeatureBuilder:
    @staticmethod
    def extract_metrics(history: List[Dict[str, Any]], target_team_id: str, client) -> Dict[str, float]:
        """Calcula promedios reales de goles, córners y remates a puerta."""
        if not history:
            return {"gf": 1.2, "ga": 1.2, "corners": 4.5, "sot": 4.0, "poss": 50.0}

        gf, ga, corners, sot, poss = [], [], [], [], []

        for m in history:
            match_id = m.get("id")
            is_home = m.get("home_team", {}).get("id") == target_team_id
            
            score = m.get("score", {})
            h_score = score.get("home", 0) or 0
            a_score = score.get("away", 0) or 0
            gf.append(h_score if is_home else a_score)
            ga.append(a_score if is_home else h_score)

            stats = client.get_match_stats(match_id)
            if stats:
                side = "home" if is_home else "away"
                t_stats = stats.get(side, {})
                corners.append(t_stats.get("corners", 0) or 0)
                sot.append(t_stats.get("shots_on_target", 0) or 0)
                poss.append(t_stats.get("possession", 50) or 50)

        return {
            "gf": float(np.mean(gf)) if gf else 1.2,
            "ga": float(np.mean(ga)) if ga else 1.2,
            "corners": float(np.mean(corners)) if corners else 4.5,
            "sot": float(np.mean(sot)) if sot else 4.0,
            "poss": float(np.mean(poss)) if poss else 50.0
        }

    @staticmethod
    def build_feature_vector(h_stats: Dict[str, float], a_stats: Dict[str, float]) -> np.ndarray:
        """Genera el vector de características de 9 dimensiones para los modelos CatBoost."""
        return np.array([[
            h_stats["gf"], h_stats["ga"], h_stats["corners"], h_stats["sot"], h_stats["poss"],
            a_stats["ga"], a_stats["corners"], a_stats["sot"], a_stats["poss"]
        ]], dtype=float)
