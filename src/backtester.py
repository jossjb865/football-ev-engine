"""Backtester post-partido para football-ev-engine.

Flujo:
  1. El scanner, al detectar que un partido ya termino, invoca
     Backtester.record_outcome() con la prediccion emitida.
  2. El backtester actualiza metricas acumuladas (Brier, calibracion,
     hit-rate, EV realizado) y persiste en JSON.
  3. Al alcanzar el umbral de muestras, invoca reentrenamiento incremental.
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

try:
    from .config import BacktestConfig
except ImportError:
    from config import BacktestConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MatchPrediction:
    """Snapshot de la prediccion emitida por el scanner."""
    match_id: str
    home_team: str
    away_team: str
    market: str
    line: float
    predicted_prob: float
    odds: float
    ev: float
    kelly_stake: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MatchOutcome:
    """Resultado real del partido para el mercado evaluado."""
    match_id: str
    market: str
    line: float
    actual_value: float
    resolved: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class BacktestMetrics:
    """Metricas acumuladas del backtester."""
    total_matches: int = 0
    hits: int = 0
    brier_sum: float = 0.0
    ev_realized_sum: float = 0.0
    ev_predicted_sum: float = 0.0

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

    Uso tipico desde scanner.py:
        bt = Backtester(cfg=bt_cfg)
        bt.record_outcome(pred, outcome)
        if bt.should_retrain():
            bt.trigger_retrain()
    """

    def __init__(self, cfg: Optional[BacktestConfig] = None):
        self.cfg = cfg or BacktestConfig()
        self.data_dir: Path = self.cfg.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.data_dir / "backtest_metrics.json"
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
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- Registro de resultados ---------------------------------------------

    def record_outcome(self, pred: MatchPrediction, outcome: MatchOutcome) -> BacktestMetrics:
        """Registra el resultado de un partido ya finalizado y actualiza metricas."""
        self._append_jsonl(self.outcomes_file, asdict(outcome))

        self.metrics.total_matches += 1

        outcome_binary = 1.0 if outcome.resolved else 0.0
        self.metrics.brier_sum += (pred.predicted_prob - outcome_binary) ** 2

        if outcome.resolved:
            self.metrics.hits += 1

        stake = pred.kelly_stake
        if outcome.resolved:
            pnl = (pred.odds - 1.0) * stake
        else:
            pnl = -stake
        self.metrics.ev_realized_sum += pnl
        self.metrics.ev_predicted_sum += pred.ev * stake

        self._save_metrics()
        logger.info(
            "Backtest: match=%s market=%s hit=%s brier=%.4f ev_realized=%.4f",
            pred.match_id, pred.market, outcome.resolved,
            self.metrics.brier_score, self.metrics.avg_ev_realized,
        )
        return self.metrics

    # -- Decision de reentrenamiento ----------------------------------------

    def should_retrain(self) -> bool:
        """True si se acumularon suficientes muestras o Brier degradado."""
        if self.metrics.total_matches >= self.cfg.retrain_threshold:
            return True
        if (self.metrics.total_matches >= self.cfg.brier_min_samples
                and self.metrics.brier_score > self.cfg.brier_degradation_threshold):
            return True
        return False

    def trigger_retrain(self) -> None:
        """Dispara reentrenamiento incremental."""
        logger.info("Backtester: umbral alcanzado (%d), disparando reentrenamiento.",
                    self.metrics.total_matches)

        try:
            from train_champions_league import retrain_incremental
        except ImportError:
            from ..train_champions_league import retrain_incremental

        outcomes_df = self._load_outcomes_df()
        retrain_incremental(
            validation_df=outcomes_df,
            brier_threshold=self.cfg.brier_degradation_threshold,
            min_samples=self.cfg.brier_min_samples,
        )

        self.metrics = BacktestMetrics()
        self._save_metrics()
        logger.info("Backtester: reentrenamiento completado, metricas reseteadas.")

    # -- Utilidades ----------------------------------------------------------

    def _load_outcomes_df(self) -> pd.DataFrame:
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
