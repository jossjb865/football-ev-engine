import os
import json
import time
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple

import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
import joblib

# Constantes oficiales de TheStatsAPI
BASE_URL = "https://api.thestatsapi.com/api"
CHAMPIONS_LEAGUE_ID = "comp_3498"
CACHE_DIR = "cache_ucl"
MODELS_DIR = "models"

API_KEY = os.getenv("THESTATSAPI_API_KEY")
if not API_KEY:
    raise ValueError("Debes configurar la variable de entorno THESTATSAPI_API_KEY.")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}


# ==========================================
# 1. CLIENTE Y DESCARGA CON CACHÉ LOCAL
# ==========================================
class TheStatsAPILoader:
    def __init__(self, api_key: str):
        self.headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        url = f"{BASE_URL}{endpoint}"
        retries = 3
        for attempt in range(retries):
            response = requests.get(url, headers=self.headers, params=params, timeout=25)
            if response.status_code == 429:
                # Rate limit alcanzado: esperar y reintentar
                time.sleep(3 * (attempt + 1))
                continue
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code} en {url}: {response.text}")
                return {}
        return {}

    def fetch_all_ucl_matches(self, start_date: str = "2020-01-01", end_date: str = "2026-10-21") -> List[Dict[str, Any]]:
        """Descarga todos los partidos finalizados de Champions League en el rango de fechas."""
        cache_file = os.path.join(CACHE_DIR, f"matches_{start_date}_{end_date}.json")
        if os.path.exists(cache_file):
            print(f"[CACHE] Cargando partidos desde {cache_file}...")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        print(f"[API] Descargando partidos de Champions League ({start_date} a {end_date})...")
        matches = []
        page = 1
        per_page = 50

        while True:
            params = {
                "competition_id": CHAMPIONS_LEAGUE_ID,
                "status": "finished",
                "date_from": start_date,
                "date_to": end_date,
                "page": page,
                "per_page": per_page
            }
            res = self._get("/football/matches", params)
            data = res.get("data", [])
            if not data:
                break

            matches.extend(data)
            meta = res.get("meta", {})
            total_pages = meta.get("total_pages", page)
            print(f"  -> Página {page}/{total_pages} descargada ({len(data)} partidos).")
            if page >= total_pages:
                break
            page += 1
            time.sleep(0.5)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(matches, f, indent=2)
        print(f"[EXITO] Total partidos obtenidos: {len(matches)}")
        return matches

    def fetch_match_stats(self, match_id: str) -> Dict[str, Any]:
        """Descarga las estadísticas detalladas (remates, córners, posesión) con caché individual."""
        cache_file = os.path.join(CACHE_DIR, f"stats_{match_id}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        res = self._get(f"/football/matches/{match_id}/stats")
        data = res.get("data", {})
        if data:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        time.sleep(0.4)  # Control de tasa (Rate limit: 120 req/min)
        return data


# ==========================================
# 2. INGENIERÍA DE VARIABLES SIN DATA LEAKAGE
# ==========================================
class FeaturePipeline:
    @staticmethod
    def process_dataset(matches: List[Dict[str, Any]], loader: TheStatsAPILoader) -> pd.DataFrame:
        """
        Ordena cronológicamente y calcula métricas móviles (Rolling Averages)
        usando exclusivamente partidos previos para cada equipo.
        """
        # Ordenar por fecha UTC ascendente
        matches = sorted(matches, key=lambda m: m.get("utc_date", ""))
        
        # Historial de partidos previos por equipo
        team_history: Dict[str, List[Dict[str, Any]]] = {}
        rows = []

        print("[PROCESAMIENTO] Extrayendo estadísticas y calculando ventanas móviles...")
        for i, m in enumerate(matches):
            match_id = m.get("id")
            utc_date = m.get("utc_date")
            home = m.get("home_team", {})
            away = m.get("away_team", {})
            home_id = home.get("id")
            away_id = away.get("id")

            if not home_id or not away_id:
                continue

            # Obtener resultado real del partido
            score = m.get("score", {})
            home_goals = score.get("home")
            away_goals = score.get("away")
            if home_goals is None or away_goals is None:
                continue

            stats = loader.fetch_match_stats(match_id)
            home_stats = stats.get("home", {})
            away_stats = stats.get("away", {})

            # Métricas del partido objetivo
            home_corners = home_stats.get("corners", 0) or 0
            away_corners = away_stats.get("corners", 0) or 0
            home_sot = home_stats.get("shots_on_target", 0) or 0
            away_sot = away_stats.get("shots_on_target", 0) or 0

            # Calcular features basadas en partidos ANTERIORES
            h_hist = team_history.get(home_id, [])
            a_hist = team_history.get(away_id, [])

            # Requerimos al menos 3 partidos de historial para computar ventanas representativas
            if len(h_hist) >= 3 and len(a_hist) >= 3:
                def get_roll_means(hist, window=5):
                    sub = hist[-window:]
                    return {
                        "gf": np.mean([x["gf"] for x in sub]),
                        "ga": np.mean([x["ga"] for x in sub]),
                        "corners_for": np.mean([x["corners_for"] for x in sub]),
                        "corners_against": np.mean([x["corners_against"] for x in sub]),
                        "sot_for": np.mean([x["sot_for"] for x in sub]),
                        "sot_against": np.mean([x["sot_against"] for x in sub]),
                        "poss": np.mean([x["poss"] for x in sub])
                    }

                h_roll = get_roll_means(h_hist, window=5)
                a_roll = get_roll_means(a_hist, window=5)

                rows.append({
                    "match_id": match_id,
                    "utc_date": utc_date,
                    # Features Equipo Local
                    "home_roll_gf": h_roll["gf"],
                    "home_roll_ga": h_roll["ga"],
                    "home_roll_corners": h_roll["corners_for"],
                    "home_roll_sot": h_roll["sot_for"],
                    "home_roll_poss": h_roll["poss"],
                    # Features Rival Visitante
                    "away_roll_ga": a_roll["ga"],
                    "away_roll_corners_against": a_roll["corners_against"],
                    "away_roll_sot_against": a_roll["sot_against"],
                    "away_roll_poss": a_roll["poss"],
                    # Targets Reales (SOLO OVERS Y BTTS)
                    "target_home_goals": int(home_goals),
                    "target_away_goals": int(away_goals),
                    "target_btts_yes": 1 if (home_goals > 0 and away_goals > 0) else 0,
                    "target_home_corners": int(home_corners),
                    "target_home_sot": int(home_sot)
                })

            # Actualizar historial para futuros partidos
            h_entry = {
                "gf": home_goals, "ga": away_goals,
                "corners_for": home_corners, "corners_against": away_corners,
                "sot_for": home_sot, "sot_against": away_sot,
                "poss": home_stats.get("possession", 50) or 50
            }
            a_entry = {
                "gf": away_goals, "ga": home_goals,
                "corners_for": away_corners, "corners_against": home_corners,
                "sot_for": away_sot, "sot_against": home_sot,
                "poss": away_stats.get("possession", 50) or 50
            }

            team_history.setdefault(home_id, []).append(h_entry)
            team_history.setdefault(away_id, []).append(a_entry)

            if (i + 1) % 50 == 0:
                print(f"  -> {i + 1}/{len(matches)} partidos procesados...")

        return pd.DataFrame(rows)


# ==========================================
# 3. MODELO DE POISSON BIVARIADA (OPTIMIZADOR)
# ==========================================
class BivariatePoissonOptimizer:
    """Ajusta paramétricamente lambda_1, lambda_2 y la covarianza lambda_3 por máxima verosimilitud."""
    def __init__(self):
        self.l1_base = 1.4
        self.l2_base = 1.1
        self.l3_cov = 0.20

    def fit(self, home_goals: np.ndarray, away_goals: np.ndarray):
        def neg_log_likelihood(params):
            l1, l2, l3 = params
            if l1 <= 0.01 or l2 <= 0.01 or l3 < 0:
                return 1e6
            log_lik = 0.0
            for x, y in zip(home_goals, away_goals):
                # Expansión truncada para Poisson Bivariada
                prob_xy = 0.0
                for k in range(min(x, y) + 1):
                    term = (math.pow(l1, x - k) / math.factorial(x - k)) * \
                           (math.pow(l2, y - k) / math.factorial(y - k)) * \
                           (math.pow(l3, k) / math.factorial(k))
                    prob_xy += term
                prob_xy *= math.exp(-(l1 + l2 + l3))
                log_lik += math.log(max(prob_xy, 1e-12))
            return -log_lik

        res = minimize(neg_log_likelihood, [1.4, 1.1, 0.2], method="Nelder-Mead", options={"maxiter": 400})
        if res.success:
            self.l1_base, self.l2_base, self.l3_cov = res.x
            print(f"  [Bivariate Poisson] Ajustado -> l1: {self.l1_base:.3f}, l2: {self.l2_base:.3f}, l3(cov): {self.l3_cov:.3f}")

    def predict_btts_prob(self) -> float:
        """P(BTTS Sí) = 1 - e^{-(l1 + l3)} - e^{-(l2 + l3)} + e^{-(l1 + l2 + l3)}"""
        p_home_zero = math.exp(-(self.l1_base + self.l3_cov))
        p_away_zero = math.exp(-(self.l2_base + self.l3_cov))
        p_both_zero = math.exp(-(self.l1_base + self.l2_base + self.l3_cov))
        return 1.0 - p_home_zero - p_away_zero + p_both_zero


# ==========================================
# 4. ORQUESTACIÓN DEL ENTRENAMIENTO
# ==========================================
def main():
    loader = TheStatsAPILoader(API_KEY)
    pipeline = FeaturePipeline()

    # 1. Obtener datos reales UCL 2020 a 2026
    raw_matches = loader.fetch_all_ucl_matches(start_date="2020-01-01", end_date="2026-10-21")
    df = pipeline.process_dataset(raw_matches, loader)

    if df.empty or len(df) < 50:
        print("[ERROR] Insuficientes datos para entrenar.")
        return

    print(f"\n[DATASET] Filas listas para modelado: {len(df)}")

    # 2. División temporal estricta (Train: 2020 - Jun 2024 / Test: Jul 2024 - 2026)
    split_date = "2024-07-01"
    train_df = df[df["utc_date"] < split_date].copy()
    test_df = df[df["utc_date"] >= split_date].copy()

    feature_cols = [
        "home_roll_gf", "home_roll_ga", "home_roll_corners", "home_roll_sot", "home_roll_poss",
        "away_roll_ga", "away_roll_corners_against", "away_roll_sot_against", "away_roll_poss"
    ]

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    print(f"  -> Set Entrenamiento: {len(train_df)} partidos")
    print(f"  -> Set Validación (Out-Of-Sample): {len(test_df)} partidos\n")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ----------------------------------------------------
    # MODELO 1: BTTS (Ambos Marcan: SÍ) - CatBoost Calibrado
    # ----------------------------------------------------
    print("--- 1. Entrenando Clasificador BTTS: Sí ---")
    y_train_btts = train_df["target_btts_yes"].values
    y_test_btts = test_df["target_btts_yes"].values

    base_btts = CatBoostClassifier(
        iterations=350,
        learning_rate=0.03,
        depth=5,
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=False
    )
    base_btts.fit(X_train, y_train_btts)

    # Calibración Isotónica para evitar sobreconfianza en cuotas
    calibrated_btts = CalibratedClassifierCV(estimator=base_btts, method="isotonic", cv="prefit")
    calibrated_btts.fit(X_train, y_train_btts)

    preds_btts_prob = calibrated_btts.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test_btts, preds_btts_prob)
    logloss_val = log_loss(y_test_btts, preds_btts_prob)
    print(f"  [Out-of-Sample] BTTS Brier Score: {brier:.4f} (Ideal < 0.24)")
    print(f"  [Out-of-Sample] BTTS Log Loss:    {logloss_val:.4f}")

    joblib.dump(calibrated_btts, os.path.join(MODELS_DIR, "ucl_btts_calibrated.joblib"))

    # ----------------------------------------------------
    # MODELO 2: Over Córners por Equipo (CatBoost Poisson)
    # ----------------------------------------------------
    print("\n--- 2. Entrenando Modelo Poisson: Over Córners Local ---")
    y_train_corners = train_df["target_home_corners"].values
    y_test_corners = test_df["target_home_corners"].values

    model_corners = CatBoostRegressor(
        iterations=400,
        learning_rate=0.03,
        depth=5,
        loss_function="Poisson",
        verbose=False
    )
    model_corners.fit(X_train, y_train_corners)
    
    pred_corners_lambda = model_corners.predict(X_test)
    mae_corners = mean_absolute_error(y_test_corners, pred_corners_lambda)
    print(f"  [Out-of-Sample] Córners MAE: {mae_corners:.2f}")

    # Evaluación de probabilidad de OVER 4.5 Córners
    line_corners = 4.5
    prob_over_45 = [float(poisson.sf(math.floor(line_corners), lmb)) for lmb in pred_corners_lambda]
    actual_over_45 = (y_test_corners > line_corners).astype(int)
    print(f"  [Out-of-Sample] Tasa Real Over 4.5 Córners:   {actual_over_45.mean()*100:.1f}%")
    print(f"  [Out-of-Sample] Tasa Media Predicha Modelo:  {np.mean(prob_over_45)*100:.1f}%")

    model_corners.save_model(os.path.join(MODELS_DIR, "ucl_home_corners_poisson.cbm"))

    # ----------------------------------------------------
    # MODELO 3: Over Remates al Arco (CatBoost Poisson)
    # ----------------------------------------------------
    print("\n--- 3. Entrenando Modelo Poisson: Over Remates al Arco Local ---")
    y_train_sot = train_df["target_home_sot"].values
    y_test_sot = test_df["target_home_sot"].values

    model_sot = CatBoostRegressor(
        iterations=400,
        learning_rate=0.03,
        depth=5,
        loss_function="Poisson",
        verbose=False
    )
    model_sot.fit(X_train, y_train_sot)

    pred_sot_lambda = model_sot.predict(X_test)
    mae_sot = mean_absolute_error(y_test_sot, pred_sot_lambda)
    print(f"  [Out-of-Sample] Remates al Arco MAE: {mae_sot:.2f}")

    # Evaluación Over 4.5 Remates al Arco
    line_sot = 4.5
    prob_sot_45 = [float(poisson.sf(math.floor(line_sot), lmb)) for lmb in pred_sot_lambda]
    actual_sot_45 = (y_test_sot > line_sot).astype(int)
    print(f"  [Out-of-Sample] Tasa Real Over 4.5 Remates:  {actual_sot_45.mean()*100:.1f}%")
    print(f"  [Out-of-Sample] Tasa Media Predicha Modelo:  {np.mean(prob_sot_45)*100:.1f}%")

    model_sot.save_model(os.path.join(MODELS_DIR, "ucl_home_sot_poisson.cbm"))

    # ----------------------------------------------------
    # MODELO 4: Poisson Bivariada de Goles (Dependencia)
    # ----------------------------------------------------
    print("\n--- 4. Ajustando Poisson Bivariada (Covarianza) ---")
    biv_opt = BivariatePoissonOptimizer()
    biv_opt.fit(train_df["target_home_goals"].values, train_df["target_away_goals"].values)
    joblib.dump(biv_opt, os.path.join(MODELS_DIR, "ucl_bivariate_poisson.joblib"))

    print("\n=======================================================")
    print(f"[COMPLETO] Modelos entrenados y guardados en '{MODELS_DIR}/'")
    print("  1. ucl_btts_calibrated.joblib")
    print("  2. ucl_home_corners_poisson.cbm")
    print("  3. ucl_home_sot_poisson.cbm")
    print("  4. ucl_bivariate_poisson.joblib")
    print("=======================================================")


if __name__ == "__main__":
    main()
