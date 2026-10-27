# football-ev-engine v2.0

Motor cuantitativo de detección de **valor esperado positivo (+EV)** en mercados
de fútbol (BTTS, goles totales, córners y remates al arco), con datos reales de
TheStatsAPI y modelos entrenados sobre Champions League.

## Arquitectura

| Módulo | Responsabilidad |
| --- | --- |
| `src/config.py` | Configuración central (umbrales, límites de cuota, Kelly) vía env vars |
| `src/api_client.py` | Cliente HTTP con reintentos, rate-limiting y caché (memoria + disco, TTL 6h) |
| `src/feature_builder.py` | Métricas de forma ponderadas por recencia, con caché de stats por partido |
| `src/models_engine.py` | Predicciones (CatBoost + Poisson bivariado), EV y stake Kelly |
| `src/odds_consolidator.py` | Mejor cuota por mercado/línea entre todas las casas (lógica pura, testeable) |
| `src/scanner.py` | Orquestación del escaneo diario e informes (MD + JSON) |
| `train_champions_league.py` | Pipeline de entrenamiento sin data leakage y división temporal estricta |

## Requisitos

- Python 3.10+
- API key de TheStatsAPI en la variable de entorno `THESTATSAPI_API_KEY`

```bash
pip install -r requirements.txt
export THESTATSAPI_API_KEY="tu_api_key"
