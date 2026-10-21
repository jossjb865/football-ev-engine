import os
import sys
from datetime import datetime, timezone
from api_client import TheStatsAPIClient
from feature_builder import FeatureBuilder
from models_engine import ModelsEngine

def main():
    target_date = os.getenv("TARGET_DATE")
    if not target_date or target_date.strip() == "":
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    min_ev_str = os.getenv("MIN_EV", "5.0")
    try:
        min_ev_threshold = float(min_ev_str) / 100.0
    except ValueError:
        min_ev_threshold = 0.05

    print(f"=======================================================")
    print(f"   SCANNER DE INVERSIÓN CUANTITATIVA (+EV) - TheStatsAPI")
    print(f"   Fecha objetivo: {target_date}")
    print(f"   Umbral de Valor Esperado: +{min_ev_threshold*100:.1f}%")
    print(f"=======================================================\n")

    client = TheStatsAPIClient()
    builder = FeatureBuilder()
    engine = ModelsEngine(models_dir="models")

    matches = client.get_fixtures(target_date)
    print(f"Partidos programados encontrados: {len(matches)}")

    picks = []

    for match in matches:
        match_id = match.get("id")
        home = match.get("home_team", {})
        away = match.get("away_team", {})
        home_name = home.get("name", "Local")
        away_name = away.get("name", "Visitante")

        if not match.get("odds_available", False):
            continue

        odds_data = client.get_match_odds(match_id)
        bookmakers = odds_data.get("bookmakers", [])
        if not bookmakers:
            continue

        # Historial de partidos previos
        h_hist = client.get_team_history(home.get("id"), limit=8)
        a_hist = client.get_team_history(away.get("id"), limit=8)

        h_stats = builder.extract_metrics(h_hist, home.get("id"), client)
        a_stats = builder.extract_metrics(a_hist, away.get("id"), client)
        features = builder.build_feature_vector(h_stats, a_stats)

        # Tasas esperadas
        lambda_goals_home = (h_stats["gf"] + a_stats["ga"]) / 2.0
        lambda_goals_away = (a_stats["gf"] + h_stats["ga"]) / 2.0
        lambda_corners_home = engine.predict_corners_lambda(features, fallback_lambda=h_stats["corners"])
        lambda_sot_home = engine.predict_sot_lambda(features, fallback_lambda=h_stats["sot"])

        match_label = f"{home_name} vs {away_name}"

        for bm in bookmakers:
            bm_name = bm.get("bookmaker", "Desconocido")
            markets = bm.get("markets", {})

            # ------------------------------------------------------------------
            # 1. MERCADO: BTTS (Ambos Equipos Marcan - EXCLUSIVAMENTE 'SÍ')
            # ------------------------------------------------------------------
            if "btts" in markets:
                odd_val = markets["btts"].get("yes", {}).get("last_seen")
                if odd_val:
                    odd = float(odd_val)
                    p_real = engine.predict_btts_probability(features, lambda_goals_home, lambda_goals_away)
                    ev = engine.calculate_ev(p_real, odd)
                    if ev >= min_ev_threshold:
                        picks.append({
                            "match": match_label,
                            "market": "BTTS: Sí",
                            "line": "-",
                            "odd": odd,
                            "p_real": p_real,
                            "ev": ev,
                            "bookmaker": bm_name
                        })

            # ------------------------------------------------------------------
            # 2. MERCADO: TOTAL GOALS (EXCLUSIVAMENTE OVERS)
            # ------------------------------------------------------------------
            if "total_goals" in markets:
                for line_str, outcomes in markets["total_goals"].items():
                    try:
                        line = float(line_str)
                    except ValueError:
                        continue
                    odd_val = outcomes.get("over", {}).get("last_seen")
                    if odd_val:
                        odd = float(odd_val)
                        p_real = engine.calculate_poisson_over(lambda_goals_home + lambda_goals_away, line)
                        ev = engine.calculate_ev(p_real, odd)
                        if ev >= min_ev_threshold:
                            picks.append({
                                "match": match_label,
                                "market": f"Over {line} Goles Totales",
                                "line": str(line),
                                "odd": odd,
                                "p_real": p_real,
                                "ev": ev,
                                "bookmaker": bm_name
                            })

            # ------------------------------------------------------------------
            # 3. MERCADO: CÓRNERS LOCAL (EXCLUSIVAMENTE OVERS)
            # ------------------------------------------------------------------
            if "corners" in markets:
                for line_str, outcomes in markets["corners"].items():
                    try:
                        line = float(line_str)
                    except ValueError:
                        continue
                    odd_val = outcomes.get("over", {}).get("last_seen")
                    if odd_val:
                        odd = float(odd_val)
                        p_real = engine.calculate_poisson_over(lambda_corners_home, line)
                        ev = engine.calculate_ev(p_real, odd)
                        if ev >= min_ev_threshold:
                            picks.append({
                                "match": match_label,
                                "market": f"Over {line} Córners ({home_name})",
                                "line": str(line),
                                "odd": odd,
                                "p_real": p_real,
                                "ev": ev,
                                "bookmaker": bm_name
                            })

            # ------------------------------------------------------------------
            # 4. MERCADO: REMATES AL ARCO LOCAL (EXCLUSIVAMENTE OVERS)
            # ------------------------------------------------------------------
            if "shots_on_target" in markets:
                for line_str, outcomes in markets["shots_on_target"].items():
                    try:
                        line = float(line_str)
                    except ValueError:
                        continue
                    odd_val = outcomes.get("over", {}).get("last_seen")
                    if odd_val:
                        odd = float(odd_val)
                        p_real = engine.calculate_poisson_over(lambda_sot_home, line)
                        ev = engine.calculate_ev(p_real, odd)
                        if ev >= min_ev_threshold:
                            picks.append({
                                "match": match_label,
                                "market": f"Over {line} Remates al Arco ({home_name})",
                                "line": str(line),
                                "odd": odd,
                                "p_real": p_real,
                                "ev": ev,
                                "bookmaker": bm_name
                            })

    # Construcción de la tabla de resultados en Markdown
    report_md = f"# 📊 Selecciones +EV Encontradas ({target_date})\n\n"
    report_md += f"**Criterio de Inversión:** Solo Overs y BTTS Sí | Valor Esperado mínimo: `+{min_ev_threshold*100:.1f}%`\n\n"

    if not picks:
        report_md += "> ⚠️ **No se detectaron oportunidades que superen el umbral de valor matemático hoy.**\n"
    else:
        report_md += "| Partido | Selección / Mercado | Cuota | Prob. Modelo | Valor Esperado (+EV) | Casa |\n"
        report_md += "| :--- | :--- | :---: | :---: | :---: | :--- |\n"
        for p in picks:
            report_md += f"| **{p['match']}** | `{p['market']}` | `{p['odd']:.2f}` | {p['p_real']*100:.1f}% | **+{p['ev']*100:.1f}%** | {p['bookmaker']} |\n"

    # Guardar en reports/
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"picks_{target_date}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Inyectar en el resumen de GitHub Actions ($GITHUB_STEP_SUMMARY)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report_md)

    print(f"\n[OK] Análisis finalizado. Picks guardados en {report_path}")

if __name__ == "__main__":
    main()
