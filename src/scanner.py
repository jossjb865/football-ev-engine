"""Scanner cuantitativo de oportunidades +EV.

Mejoras respecto a la version original:
- CLI con argparse (--date, --min-ev, --models-dir, --out, --no-cache, --skip-backtest).
- Logging estructurado en lugar de prints sueltos.
- Cache de historiales por equipo y de estadisticas por partido (FeatureBuilder).
- La consolidacion de cuotas es tolerante a datos malformados (no se cae).
- Informe en Markdown + JSON, picks ordenados por EV descendente.
- Stake sugerida con Kelly fraccional.
- Backtesting post-partido: al detectar partidos finalizados, registra
  prediccion vs. resultado y dispara reentrenamiento si la calibracion
  se degrada.
- Compatible con ejecucion como paquete (python -m src.scanner) o directa.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

try:
    from .api_client import APIError, TheStatsAPIClient
    from .backtester import Backtester, MatchPrediction, MatchOutcome
    from .config import ScannerConfig, BacktestConfig
    from .feature_builder import FeatureBuilder
    from .models_engine import ModelsEngine
    from .odds_consolidator import consolidate
except ImportError:  # ejecucion directa: python src/scanner.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from api_client import APIError, TheStatsAPIClient
    from backtester import Backtester, MatchPrediction, MatchOutcome
    from config import ScannerConfig, BacktestConfig
    from feature_builder import FeatureBuilder
    from models_engine import ModelsEngine
    from odds_consolidator import consolidate

log = logging.getLogger("ev-scanner")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scanner cuantitativo +EV (lineas unicas y consolidadas)")
    parser.add_argument("--date", default=os.getenv("TARGET_DATE", ""),
                        help="Fecha a analizar YYYY-MM-DD (por defecto: hoy UTC).")
    parser.add_argument("--min-ev", type=float, default=None,
                        help="Umbral minimo de EV en %% (default: env MIN_EV o 5.0).")
    parser.add_argument("--models-dir", default=os.getenv("MODELS_DIR"),
                        help="Directorio con los modelos entrenados.")
    parser.add_argument("--out", default="reports",
                        help="Directorio de salida de informes.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Desactiva el cache en disco de la API.")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Omite la fase de backtesting post-partido.")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------
# Backtesting helpers
# ----------------------------------------------------------------------
def _resolve_market(match: dict, market_label: str, line: float) -> bool:
    """Determina si el mercado resolvio a favor de la prediccion (Over/Si)."""
    score = match.get("score", {})
    home_goals = score.get("home", 0)
    away_goals = score.get("away", 0)
    total_goals = home_goals + away_goals

    if "BTTS" in market_label:
        return home_goals > 0 and away_goals > 0

    try:
        line_val = float(line)
    except (ValueError, TypeError):
        return False

    if "Corners" in market_label:
        total_corners = score.get("home_corners", 0) + score.get("away_corners", 0)
        return total_corners > line_val
    if "Remates" in market_label:
        total_sot = score.get("home_sot", 0) + score.get("away_sot", 0)
        return total_sot > line_val

    return total_goals > line_val


def _run_backtest(client: TheStatsAPIClient, bt_cfg: BacktestConfig,
                  target_date: str) -> dict:
    """
    Fase de backtesting: detecta partidos finalizados de la fecha objetivo,
    recupera las predicciones previas y registra outcomes.
    """
    bt = Backtester(cfg=bt_cfg)
    report = bt.get_report()

    pred_file = bt.data_dir / f"predictions_{target_date}.jsonl"
    if not pred_file.exists():
        log.info("Backtest: sin predicciones previas para %s, se omite.", target_date)
        return report

    predictions = [json.loads(l) for l in pred_file.read_text().splitlines() if l.strip()]
    log.info("Backtest: %d predicciones previas para %s", len(predictions), target_date)

    try:
        matches = client.get_fixtures(target_date)
    except APIError as exc:
        log.warning("Backtest: no se pudieron obtener resultados: %s", exc)
        return report

    finished = [m for m in matches if m.get("status") in ("finished", "AET", "PEN")]
    log.info("Backtest: %d partidos finalizados de %d", len(finished), len(matches))

    for match in finished:
        match_id = match.get("id")
        for pred in predictions:
            if pred.get("match_id") != match_id:
                continue

            market_label = pred.get("market", "")
            line = pred.get("line", 0.0)
            resolved = _resolve_market(match, market_label, line)

            outcome = MatchOutcome(
                match_id=match_id,
                market=market_label,
                line=line,
                actual_value=match.get("score", {}).get("home", 0)
                             + match.get("score", {}).get("away", 0),
                resolved=resolved,
            )

            m_pred = MatchPrediction(
                match_id=match_id,
                home_team=pred.get("home_team", ""),
                away_team=pred.get("away_team", ""),
                market=market_label,
                line=line,
                predicted_prob=pred.get("p_real", 0.5),
                odds=pred.get("odd", 2.0),
                ev=pred.get("ev", 0.0),
                kelly_stake=pred.get("stake", 0.0),
            )

            bt.record_outcome(m_pred, outcome)

    if bt.should_retrain():
        bt.trigger_retrain()

    return bt.get_report()


# ----------------------------------------------------------------------
# Informe
# ----------------------------------------------------------------------
def render_markdown(picks: list, target_date: str, min_ev_pct: float,
                    backtest_report: dict | None = None) -> str:
    md = f"# 📊 Selecciones +EV Optimizadas ({target_date})\n\n"
    md += (f"**Criterio:** Línea única por mercado | Solo Overs puros (.5) y BTTS Sí "
           f"| Cuotas 1.40 a 3.80 | EV ≥ +{min_ev_pct:.1f}%\n\n")
    if not picks:
        md += ("> ⚠️ **No se detectaron oportunidades con valor "
               "matemático suficiente para hoy.**\n\n")
    else:
        md += ("| Partido | Selección / Mercado | Mejor Cuota | Prob. Real | "
               "EV | Stake (Kelly ¼) | Casa |\n")
        md += ("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for p in picks:
            md += (f"| **{p['match']}** | `{p['market']}` | `{p['odd']:.2f}` "
                   f"| {p['p_real'] * 100:.1f}% | **+{p['ev'] * 100:.1f}%** "
                   f"| {p['stake'] * 100:.2f}% | {p['bookmaker']} |\n")

    if backtest_report and backtest_report.get("total_matches", 0) > 0:
        md += ("\n## 📈 Backtesting\n\n")
        md += (f"| Métrica | Valor |\n| :--- | :---: |\n")
        md += (f"| Partidos evaluados | {backtest_report['total_matches']} |\n")
        md += (f"| Hit rate | {backtest_report['hit_rate'] * 100:.1f}% |\n")
        md += (f"| Brier score | {backtest_report['brier_score']:.4f} |\n")
        md += (f"| EV promedio predicho | {backtest_report['avg_ev_predicted'] * 100:+.2f}% |\n")
        md += (f"| EV promedio realizado | {backtest_report['avg_ev_realized'] * 100:+.2f}% |\n")
        if backtest_report.get("should_retrain"):
            md += ("\n> 🔁 **Reentrenamiento pendiente** (umbral alcanzado).\n")

    return md


# ----------------------------------------------------------------------
# Persistencia de predicciones (para backtesting en la siguiente corrida)
# ----------------------------------------------------------------------
def _save_predictions_for_backtest(picks: list, matches: list,
                                   target_date: str, bt_cfg: BacktestConfig) -> None:
    """Guarda las predicciones emitidas para que el backtester las use luego."""
    bt_dir = Path(bt_cfg.data_dir)
    bt_dir.mkdir(parents=True, exist_ok=True)
    pred_file = bt_dir / f"predictions_{target_date}.jsonl"

    match_map = {f"{m.get('home_team', {}).get('name', '')} vs "
                 f"{m.get('away_team', {}).get('name', '')}": m.get("id")
                 for m in matches}

    records = []
    for p in picks:
        match_id = match_map.get(p["match"], "")
        line = 0.0
        parts = p["market"].split()
        for i, part in enumerate(parts):
            if part == "Over" and i + 1 < len(parts):
                try:
                    line = float(parts[i + 1])
                except ValueError:
                    pass
                break

        records.append({
            "match_id": match_id,
            "home_team": p["match"].split(" vs ")[0],
            "away_team": p["match"].split(" vs ")[1] if " vs " in p["match"] else "",
            "market": p["market"],
            "line": line,
            "p_real": p["p_real"],
            "odd": p["odd"],
            "ev": p["ev"],
            "stake": p["stake"],
        })

    with open(pred_file, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("Backtest: %d predicciones guardadas en %s", len(records), pred_file)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = ScannerConfig()
    bt_cfg = BacktestConfig()
    if args.min_ev is not None:
        cfg = replace(cfg, min_ev=args.min_ev / 100.0)
    target_date = args.date.strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    log.info("Fecha: %s | EV minimo: +%.1f%%", target_date, cfg.min_ev * 100)

    client = TheStatsAPIClient(
        cache_dir=None if args.no_cache else ".cache_api",
        cache_ttl=6 * 3600,
    )
    builder = FeatureBuilder(half_life=cfg.recency_half_life)
    engine = ModelsEngine(models_dir=args.models_dir)

    try:
        matches = client.get_fixtures(target_date)
    except APIError as exc:
        log.error("No se pudieron obtener los partidos: %s", exc)
        return 1
    log.info("Partidos programados encontrados: %d", len(matches))

    history_cache: dict = {}

    def history_of(team_id, limit: int) -> list:
        if team_id not in history_cache:
            try:
                history_cache[team_id] = client.get_team_history(team_id, limit=limit)
            except APIError as exc:
                log.warning("Historial no disponible para %s: %s", team_id, exc)
                history_cache[team_id] = []
        return history_cache[team_id]

    picks: list = []

    for match in matches:
        match_id = match.get("id")
        home = match.get("home_team", {})
        away = match.get("away_team", {})
        home_name = home.get("name", "Local")
        away_name = away.get("name", "Visitante")
        label = f"{home_name} vs {away_name}"

        if not match.get("odds_available", False):
            continue

        try:
            odds_data = client.get_match_odds(match_id)
        except APIError as exc:
            log.warning("Cuotas no disponibles para %s: %s", label, exc)
            continue
        bookmakers = odds_data.get("bookmakers", [])
        if not bookmakers:
            continue

        h_stats = builder.extract_metrics(
            history_of(home.get("id"), cfg.history_limit), home.get("id"), client)
        a_stats = builder.extract_metrics(
            history_of(away.get("id"), cfg.history_limit), away.get("id"), client)
        features = builder.build_feature_vector(h_stats, a_stats)

        lam_h = (h_stats["gf"] + a_stats["ga"]) / 2.0
        lam_a = (a_stats["gf"] + h_stats["ga"]) / 2.0
        lam_c = engine.predict_corners_lambda(features, h_stats["corners"])
        lam_s = engine.predict_sot_lambda(features, h_stats["sot"])

        best = consolidate(bookmakers, cfg)

        groups = {
            "btts": ([("BTTS: Sí",
                       engine.predict_btts_probability(features, lam_h, lam_a),
                       best["btts_yes"])]
                     if best["btts_yes"]["odd"] > 0 else []),
            "total_goals": [
                (f"Over {line} Goles Totales",
                 engine.calculate_poisson_over(lam_h + lam_a, line), data)
                for line, data in best["total_goals"].items()],
            "corners": [
                (f"Over {line} Córners ({home_name})",
                 engine.calculate_poisson_over(lam_c, line), data)
                for line, data in best["corners"].items()],
            "shots_on_target": [
                (f"Over {line} Remates al Arco ({home_name})",
                 engine.calculate_poisson_over(lam_s, line), data)
                for line, data in best["shots_on_target"].items()],
        }

        for candidates in groups.values():
            best_pick = None
            for market_label, prob, data in candidates:
                ev = engine.calculate_ev(prob, data["odd"])
                if ev >= cfg.min_ev and (best_pick is None or ev > best_pick["ev"]):
                    best_pick = {
                        "match": label,
                        "market": market_label,
                        "odd": data["odd"],
                        "p_real": prob,
                        "ev": ev,
                        "stake": engine.kelly_stake(
                            prob, data["odd"], cfg.kelly_fraction),
                        "bookmaker": data["bookmaker"],
                    }
            if best_pick:
                picks.append(best_pick)

    picks.sort(key=lambda p: p["ev"], reverse=True)

    # ---- Backtesting post-partido ----
    backtest_report = None
    if not args.skip_backtest:
        backtest_report = _run_backtest(client, bt_cfg, target_date)

    # ---- Guardar predicciones para backtesting futuro ----
    _save_predictions_for_backtest(picks, matches, target_date, bt_cfg)

    # ---- Salida: JSON + Markdown ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"picks_{target_date}.json").write_text(
        json.dumps(picks, indent=2, ensure_ascii=False), encoding="utf-8")

    report_md = render_markdown(picks, target_date, cfg.min_ev * 100, backtest_report)
    (out_dir / f"picks_{target_date}.md").write_text(report_md, encoding="utf-8")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report_md)

    log.info("Analisis finalizado. Total picks unicos de valor: %d", len(picks))
    print(report_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())   
