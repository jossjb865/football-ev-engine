"""Cliente HTTP para TheStatsAPI con reintentos, rate-limiting y cache.

Mejoras respecto a la version original:
- Reutiliza conexion mediante ``requests.Session``.
- Backoff exponencial con jitter y soporte del header ``Retry-After``.
- Cache en memoria + disco (TTL configurable) para no repetir llamadas
  identicas entre corridas (historial de equipos, estadisticas de partidos).
- Espera minima entre peticiones para respetar el rate limit de la API.
- Errores explicitos (``APIError``) en lugar de dicts vacios que
  enmascaraban fallos.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class APIError(RuntimeError):
    """Error definitivo tras agotar los reintentos contra la API."""


class TheStatsAPIClient:
    BASE_URL = "https://api.thestatsapi.com/api"
    MAX_RETRIES = 4
    TIMEOUT = 20

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 6 * 3600,
        min_interval: float = 0.55,
    ):
        self.api_key = api_key or os.getenv("THESTATSAPI_API_KEY")
        if not self.api_key:
            raise ValueError("Variable de entorno THESTATSAPI_API_KEY no encontrada.")
        self.cache_ttl = cache_ttl
        self.min_interval = min_interval
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })
        self._last_request_ts = 0.0
        self._mem_cache: Dict[str, tuple] = {}  # key -> (timestamp, payload)
        self._disk_dir = Path(cache_dir) if cache_dir else None
        if self._disk_dir:
            self._disk_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _cache_key(self, endpoint: str, params: Optional[Dict[str, Any]]) -> str:
        raw = json.dumps({"e": endpoint, "p": params or {}}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        hit = self._mem_cache.get(key)
        if hit and now - hit[0] < self.cache_ttl:
            return hit[1]
        if self._disk_dir:
            f = self._disk_dir / f"{key}.json"
            if f.exists() and now - f.stat().st_mtime < self.cache_ttl:
                try:
                    payload = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
                self._mem_cache[key] = (now, payload)
                return payload
        return None

    def _cache_set(self, key: str, payload: Dict[str, Any]) -> None:
        now = time.time()
        self._mem_cache[key] = (now, payload)
        if self._disk_dir:
            try:
                (self._disk_dir / f"{key}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            except OSError:
                pass  # el cache de disco es un optimo, nunca un fallo

    # ------------------------------------------------------------------
    # HTTP con reintentos
    # ------------------------------------------------------------------
    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
             use_cache: bool = True) -> Dict[str, Any]:
        key = self._cache_key(endpoint, params)
        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        url = f"{self.BASE_URL}{endpoint}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            self._respect_rate_limit()
            try:
                response = self._session.get(url, params=params, timeout=self.TIMEOUT)
                self._last_request_ts = time.time()
            except requests.RequestException as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 429:  # rate limited por la API
                time.sleep(self._retry_after(response, attempt))
                continue
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    last_exc = exc
                    self._sleep_backoff(attempt)
                    continue
                if use_cache:
                    self._cache_set(key, payload)
                return payload
            if 500 <= response.status_code < 600:  # error transitorio del servidor
                last_exc = APIError(f"HTTP {response.status_code} en {url}")
                self._sleep_backoff(attempt)
                continue
            # 4xx definitivo: no reintentar
            raise APIError(f"HTTP {response.status_code} en {url}: {response.text[:200]}")

        raise APIError(f"Fallo tras {self.MAX_RETRIES} reintentos en {url}: {last_exc}")

    @staticmethod
    def _retry_after(response: requests.Response, attempt: int) -> float:
        try:
            header = float(response.headers.get("Retry-After", 0))
        except (TypeError, ValueError):
            header = 0.0
        backoff = min(2 ** attempt, 16) + random.random()
        return max(header, backoff)

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        time.sleep(min(2 ** attempt, 16) + random.random())

    # ------------------------------------------------------------------
    # Endpoints de alto nivel
    # ------------------------------------------------------------------
    def get_fixtures(self, date_str: str) -> List[Dict[str, Any]]:
        """Partidos programados con cuotas para la fecha indicada."""
        data = self._get("/football/matches", {
            "date_from": date_str,
            "date_to": date_str,
            "status": "scheduled",
        })
        return data.get("data", [])

    def get_match_odds(self, match_id: str) -> Dict[str, Any]:
        """Cuotas prepartido de todas las casas disponibles."""
        data = self._get(f"/football/matches/{match_id}/odds")
        return data.get("data", {})

    def get_team_history(self, team_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Ultimos partidos finalizados de un equipo."""
        data = self._get("/football/matches", {
            "team_id": team_id,
            "status": "finished",
            "per_page": limit,
        })
        return data.get("data", [])

    def get_match_stats(self, match_id: str) -> Dict[str, Any]:
        """Estadisticas avanzadas de remates, corners y posesion."""
        data = self._get(f"/football/matches/{match_id}/stats")
        return data.get("data", {})
