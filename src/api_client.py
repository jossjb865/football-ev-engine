import os
import time
from typing import Dict, Any, List, Optional
import requests

class TheStatsAPIClient:
    BASE_URL = "https://api.thestatsapi.com/api"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("THESTATSAPI_API_KEY")
        if not self.api_key:
            raise ValueError("Variable de entorno THESTATSAPI_API_KEY no encontrada.")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{endpoint}"
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=20)
                if response.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                if response.status_code == 200:
                    return response.json()
                return {}
            except requests.RequestException:
                time.sleep(1)
        return {}

    def get_fixtures(self, date_str: str) -> List[Dict[str, Any]]:
        """Devuelve los partidos programados con cuotas para la fecha indicada."""
        data = self._get("/football/matches", {
            "date_from": date_str,
            "date_to": date_str,
            "status": "scheduled"
        })
        return data.get("data", [])

    def get_match_odds(self, match_id: str) -> Dict[str, Any]:
        """Obtiene las cuotas prepartido de todas las casas disponibles."""
        data = self._get(f"/football/matches/{match_id}/odds")
        return data.get("data", {})

    def get_team_history(self, team_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene los últimos partidos finalizados de un equipo."""
        data = self._get("/football/matches", {
            "team_id": team_id,
            "status": "finished",
            "per_page": limit
        })
        return data.get("data", [])

    def get_match_stats(self, match_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas avanzadas de remates, córners y posesión."""
        data = self._get(f"/football/matches/{match_id}/stats")
        return data.get("data", {})
