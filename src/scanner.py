import os
import sys
from datetime import datetime, timezone
from collections import defaultdict
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
    print(f"   SCANNER CUANTITATIVO +EV (Líneas Únicas y Consolidadas)")
    print(f"   Fecha: {target_date} | EV Mínimo: +{min_ev_threshold*100:.1f}%")
    print(f"=======================================================\n")

    client = TheStatsAPIClient()
    builder = FeatureBuilder()
    engine = ModelsEngine(models_dir="models")

    matches = client.get_fixtures(target_date)
    print(f"Partidos programados encontrados: {len(matches)}")

    final_picks = []

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

        # Historial de métricas previas
        h_hist = client.get_team_history(home.get("id"), limit=8)
        a_hist = client.get_team_history(away.get("id"), limit=8)

        h_stats = builder.extract_metrics(h_hist, home.get("id"), client)
        a_stats = builder.extract_metrics(a_hist, away.get("id"), client)
        features = builder.build_feature_vector(h_stats, a_stats)

        # Tasas esperadas Poisson
        lambda_goals_home = (h_stats["gf"] + a_stats["ga"]) / 2.0
        lambda_goals_away = (a_stats["gf"] + h_stats["ga"]) / 2.0
        lambda_corners_home = engine.predict_corners_lambda(features, fallback_lambda=h_stats["corners"])
        lambda_sot_home = engine.predict_sot_lambda(features, fallback_lambda=h_stats["sot"])

        match_label = f"{home_name} vs {away_name}"

        # Consolidar la MEJOR cuota por mercado/línea entre todas las bookmakers
        # Estructura: best_odds[mercado_key] = {"odd": max_odd, "bookmaker": nombre}
        best_btts_yes = {"odd": 0.0, "bookmaker": ""}
        best_total_goals = defaultdict(lambda: {"odd": 0.0, "bookmaker": ""})
        best_corners = defaultdict(lambda: {"odd": 0.0, "bookmaker": ""})
        best_sot = defaultdict(lambda: {"odd": 0.0, "bookmaker": ""})

        for bm in bookmakers:
            bm_name = bm.get("bookmaker", "Desconocido")
            markets = bm.get("markets", {})

            # 1. BTTS: Sí
            if "btts" in markets:
                odd_str = markets["btts"].get("yes", {}).get("last_seen")
                if odd_str:
                    odd = float(odd_str)
                    if odd > best_btts_yes["odd"]:
                        best_btts_yes = {"odd": odd, "bookmaker": bm_name}

            # 2. Total Goles (Solo líneas puras .5 y cuotas de 1.40 a 3.80)
            if "total_goals" in markets:
                for line_str, outcomes in markets["total_goals"].items():
                    try:
                        line = float(line_str)
                    except ValueError:
                        continue
                    # REGLA: Excluir asiáticos (.25, .75) y enteros. Solo .5
                    if line % 1.0 != 0.5:
                        continue
                    odd_str = outcomes.get("over", {}).get("last_seen")
                    if odd_str:
                        odd = float(odd_str)
                        if 1.40 <= odd <= 3.80 and odd > best_total_goals[line]["odd"]:
                            best_total_goals[line] = {"odd": odd, "bookmaker": bm_name}

            # 3. Córners Local (Solo líneas puras .5 y cuotas de 1.40 a 3.80)
            if "corners" in markets:
                for line_str, outcomes in markets["corners"].items():
                    try:
                        line = float(line_str)
                    except ValueError:
                        continue
                    if line % 1.0 != 0.5:
                        continue
                    odd_str = outcomes.get("over", {}).get("last_seen")
                    if odd_str:
                        odd = float(odd_str)
                        if 1.40 <= odd <= 3.80 and odd > best_corners[line]["odd"]:
                            best_corners[line] = {"odd": odd, "bookmaker": bm_name}

            # 4. Remates al Arco Local (Solo líneas puras .5)
            if "shots_on_target" in markets:
                for line_str, outcomes in markets["shots_on_target"].items():
                    try:
                        line = float(line_str)
                    except ValueError:
                        continue
                    if line % 1.0 != 0.5:
                        continue
                    odd_str = outcomes.get("over", {}).get("last_seen")
                    if odd_str:
                        odd = float(odd_str)
                        if 1.40 <= odd <= 3.80 and odd > best_sot[line]["odd"]:
                            best_sot[line] = {"odd": odd, "bookmaker": bm_name}

        # --- EVALUACIÓN Y SELECCIÓN DEL PICK ÓPTIMO POR MERCADO ---

        # Evaluar BTTS: Sí
        if best_btts_yes["odd"] > 0:
            p_real = engine.predict_btts_probability(features, lambda_goals_home, lambda_goals_away)
            ev = engine.calculate_ev(p_real, best_btts_yes["odd"])
            if ev >= min_ev_threshold:
                final_picks.append({
                    "match": match_label,
                    "market": "BTTS: Sí",
                    "odd": best_btts_yes["odd"],
                    "p_real": p_real,
                    "ev": ev,
                    "bookmaker": best_btts_yes["bookmaker"]
                })

        # Evaluar Total Goles: SELECCIONAR ÚNICAMENTE LA MEJOR LÍNEA (Mayor EV)
        best_goals_pick = None
        for line, data in best_total_goals.items():
            p_real = engine.calculate_poisson_over(lambda_goals_home + lambda_goals_away, line)
            ev = engine.calculate_ev(p_real, data["odd"])
            if ev >= min_ev_threshold:
                candidate = {
                    "match": match_label,
                    "market": f"Over {line} Goles Totales",
                    "odd": data["odd"],
                    "p_real": p_real,
                    "ev": ev,
                    "bookmaker": data["bookmaker"]
                }
                if best_goals_pick is None or ev > best_goals_pick["ev"]:
                    best_goals_pick = candidate
        if best_goals_pick:
            final_picks.append(best_goals_pick)

        # Evaluar Córners: SELECCIONAR ÚNICAMENTE LA MEJOR LÍNEA
        best_corners_pick = None
        for line, data in best_corners.items():
            p_real = engine.calculate_poisson_over(lambda_corners_home, line)
            ev = engine.calculate_ev(p_real, data["odd"])
            if ev >= min_ev_threshold:
                candidate = {
                    "match": match_label,
                    "market": f"Over {line} Córners ({home_name})",
                    "odd": data["odd"],
                    "p_real": p_real,
                    "ev": ev,
                    "bookmaker": data["bookmaker"]
                }
                if best_corners_pick is None or ev > best_corners_pick["ev"]:
                    best_corners_pick = candidate
        if best_corners_pick:
            final_picks.append(best_corners_pick)

        # Evaluar Remates al Arco: SELECCIONAR ÚNICAMENTE LA MEJOR LÍNEA
        best_sot_pick = None
        for line, data in best_sot.items():
            p_real = engine.calculate_poisson_over(lambda_sot_home, line)
            ev = engine.calculate_ev(p_real, data["odd"])
            if ev >= min_ev_threshold:
                candidate = {
                    "match": match_label,
                    "market": f"Over {line} Remates al Arco ({home_name})",
                    "odd": data["odd"],
                    "p_real": p_real,
                    "ev": ev,
                    "bookmaker": data["bookmaker"]
                }
                if best_sot_pick is None or ev > best_sot_pick["ev"]:
                    best_sot_pick = candidate
        if best_sot_pick:
            final_picks.append(best_sot_pick)

    # Generación de la tabla final limpia
    report_md = f"# 📊 Selecciones +EV Optimizadas ({target_date})\n\n"
    report_md += f"**Criterio:** Línea única por mercado | Solo Overs puros (.5) y BTTS Sí | Cuotas 1.40 a 3.80 | EV $\\ge +{min_ev_threshold*100:.1f}\\%$\n\n"

    if not final_picks:
        report_md += "> ⚠️ **No se detectaron oportunidades con valor matemático suficiente para hoy.**\n"
    else:
        report_md += "| Partido | Selección / Mercado | Mejor Cuota | Prob. Real | Valor Esperado (+EV) | Casa |\n"
        report_md += "| :--- | :--- | :---: | :---: | :---: | :--- |\n"
        for p in final_picks:
            report_md += f"| **{p['match']}** | `{p['market']}` | `{p['odd']:.2f}` | {p['p_real']*100:.1f}% | **+{p['ev']*100:.1f}%** | {p['bookmaker']} |\n"

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"picks_{target_date}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report_md)

    print(f"\n[OK] Análisis finalizado. Total picks únicos de valor: {len(final_picks)}")

if __name__ == "__main__":
    main()
