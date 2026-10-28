"""
Backtester post-partido para football-ev-engine.

Flujo:
  1. El scanner, al detectar que un partido ya terminó, invoca
     Backtester.record_outcome() con la predicción emitida.
  2. El backtester actualiza métricas acumuladas (Brier, calibración,
     hit-rate, EV realizado) y persiste en JSON.
  3. Al alcanzar BACKTEST_RETRAIN_THRESHOLD muestras, invoca
     train_champions_league.retrain_incremental() para actualizar
     los pesos del modelo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import BACKTEST_DATA_DIR, BACKTEST_RETRAIN_THRESHOLD, EV_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MatchPrediction:
    """Snapshot de la predicción emitida por el scanner."""
    match_id: str
    home_team: str
    away_team: str
    market: str            # "btts", "total_goals", "corners", "shots_on_target"
    line: float            # e.g. 2.5 para total goals
    predicted_prob: float  # probabilidad del modelo (0-1)
    odds: float            # mejor cuota consolidada
    ev: float              # EV calculado por models_engine
    kelly_stake: float     # fracción Kelly
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MatchOutcome:
    """Resultado real del partido para el mercado evaluado."""
    match_id: str
    market: str
    line: float
    actual_value: float    # e.g. goles totales = 3.0, BTTS = 1 (sí) / 0 (no)
    resolved: bool         # True si el mercado resolvió a favor de la predicción
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class BacktestMetrics:
    """Métricas acumuladas del backtester."""
    total_matches: int = 0
    hits: int = 0
    brier_sum: float = 0.0
    ev_realized_sum: float = 0.0
    ev_predicted_sum: float = 0.0
    calibration_bins: list = field(default_factory=list)

    @property
    def brier_score(self) -> float:
        return self.brier_sum / self.total_matches if self.total_matches else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_matches if self.total_matches else 0.0

    @property
    def avg_ev_realized(self) -> float:
        return self.ev_realized_sum / self.total_matches if self.total_matches else 0.0


# ---------------------------------------------------------------------------
# Backtester principal
# ---------------------------------------------------------------------------

class Backtester:
    """
    Motor de backtesting post-partido.

    Uso típico desde scanner.py:
        bt = Backtester()
        bt.record_outcome(pred, outcome)
        if bt.should_retrain():
            bt.trigger_retrain()
    """

    def __init__(self, data_dir: Path = BACKTEST_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.data_dir / "backtest_metrics.json"
        self.predictions_file = self.data_dir / "predictions.jsonl"
        self.outcomes_file = self.data_dir / "outcomes.jsonl"
        self.metrics: BacktestMetrics = self._load_metrics()

    # -- Persistencia -------------------------------------------------------

    def _load_metrics(self) -> BacktestMetrics:
        if self.metrics_file.exists():
            data = json.loads(self.metrics_file.read_text())
            return BacktestMetrics(**data)
        return BacktestMetrics()

    def _save_metrics(self) -> None:
        self.metrics_file.write_text(json.dumps(asdict(self.metrics), indent=2))

    def _append_jsonl(self, path: Path, record: dict) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # -- Registro de resultados ---------------------------------------------

    def record_outcome(self, pred: MatchPrediction, outcome: MatchOutcome) -> BacktestMetrics:
        """
        Registra el resultado de un partido ya finalizado y actualiza métricas.

        Parameters
        ----------
        pred : MatchPrediction
            La predicción que emitió el scanner antes del partido.
        outcome : MatchOutcome
            El resultado real del mercado.
        """
        # 1. Persistir predicción y resultado
        self._append_jsonl(self.predictions_file, asdict(pred))
        self._append_jsonl(self.outcomes_file, asdict(outcome))

        # 2. Actualizar métricas
        self.metrics.total_matches += 1

        # Brier score: (prob - outcome_binary)²
        outcome_binary = 1.0 if outcome.resolved else 0.0
        self.metrics.brier_sum += (pred.predicted_prob - outcome_binary) ** 2

        # Hit rate
        if outcome.resolved:
            self.metrics.hits += 1

        # EV realizado: si acertó, gana (odds-1)*stake; si falló, -stake
        stake = pred.kelly_stake
        if outcome.resolved:
            pnl = (pred.odds - 1.0) * stake
        else:
            pnl = -stake
        self.metrics.ev_realized_sum += pnl
        self.metrics.ev_predicted_sum += pred.ev * stake

        # 3. Persistir
        self._save_metrics()
        logger.info(
            "Backtest: match=%s market=%s hit=%s brier=%.4f ev_realized=%.4f",
            pred.match_id, pred.market, outcome.resolved,
            self.metrics.brier_score, self.metrics.avg_ev_realized,
        )
        return self.metrics

    # -- Decisión de reentrenamiento ----------------------------------------

    def should_retrain(self) -> bool:
        """True si se acumularon suficientes muestras para reentrenar."""
        return self.metrics.total_matches >= BACKTEST_RETRAIN_THRESHOLD

    def trigger_retrain(self) -> None:
        """
        Dispara reentrenamiento incremental usando los datos del backtest
        como validación out-of-sample.
        """
        logger.info("Backtester: umbral alcanzado (%d), disparando reentrenamiento.",
                    self.metrics.total_matches)

        # Import dinámico para evitar circular import
        from train_champions_league import retrain_incremental

        # Cargar outcomes como dataset de validación
        outcomes_df = self._load_outcomes_df()
        retrain_incremental(
            validation_df=outcomes_df,
            brier_threshold=0.25,          # re-entrenar si Brier > 0.25
            min_samples=BACKTEST_RETRAIN_THRESHOLD,
        )

        # Resetear contador para el siguiente ciclo
        self.metrics.total_matches = 0
        self.metrics.hits = 0
        self.metrics.brier_sum = 0.0
        self.metrics.ev_realized_sum = 0.0
        self.metrics.ev_predicted_sum = 0.0
        self._save_metrics()
        logger.info("Backtester: reentrenamiento completado, métricas reseteadas.")

    # -- Utilidades ----------------------------------------------------------

    def _load_outcomes_df(self) -> pd.DataFrame:
        """Carga outcomes acumulados como DataFrame para reentrenamiento."""
        if not self.outcomes_file.exists():
            return pd.DataFrame()
        records = [json.loads(line) for line in self.outcomes_file.read_text().splitlines() if line.strip()]
        return pd.DataFrame(records)

    def get_report(self) -> dict:
        """Genera informe de backtesting para el informe MD/JSON del scanner."""
        return {
            "total_matches": self.metrics.total_matches,
            "hit_rate": round(self.metrics.hit_rate, 4),
            "brier_score": round(self.metrics.brier_score, 4),
            "avg_ev_predicted": round(
                self.metrics.ev_predicted_sum / max(self.metrics.total_matches, 1), 4
            ),
            "avg_ev_realized": round(self.metrics.avg_ev_realized, 4),
            "should_retrain": self.should_retrain(),
        }   
