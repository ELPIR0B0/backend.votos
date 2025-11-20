# Backend: Servicio KNN de intención de voto

API REST en FastAPI que expone el modelo K-Nearest Neighbors entrenado sobre el dataset `voter_intentions_3000.csv`. El flujo completo es:

Notebook en Colab → genera `models/knn_voter_intention_pipeline.pkl` → este backend carga el artefacto → expone predicciones vía HTTP.

## Requisitos

- Python 3.11+
- Dependencias en `requirements.txt`
- Archivo de modelo en `models/knn_voter_intention_pipeline.pkl` (generado con el notebook `notebooks/knn_voter_intention.ipynb`)

## Estructura

- `app/main.py`: FastAPI + endpoints (`/health`, `/model-info`, `/predict`, `/predict-batch`)
- `app/ml/model.py`: carga del pipeline y predicción
- `app/schemas/prediction.py`: esquemas Pydantic
- `app/config/settings.py`: configuración via variables de entorno
- `models/`: contiene el artefacto `.pkl`
- `notebooks/knn_voter_intention.ipynb`: notebook de entrenamiento listo para Colab

## Variables de entorno

- `MODEL_PATH` (opcional): ruta al archivo `.pkl`. Por defecto `models/knn_voter_intention_pipeline.pkl`.
- `API_TITLE`, `API_VERSION`, `DEBUG` (opcional).

## Cómo correr en local

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate        # en Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Visita `http://localhost:8000/docs` para la UI interactiva.

## Endpoints principales

- `GET /health` → `{ "status": "ok" }`
- `GET /model-info` → metadatos del modelo (k, métrica, clases, columnas de entrada, etc.)
- `POST /predict` → cuerpo JSON con las 32 columnas de entrada (todas opcionales, el pipeline imputa faltantes). Devuelve `predicted_vote`, probabilidades por clase y vecinos cercanos aproximados.
- `POST /predict-batch` → lista de objetos como el de `/predict`.

## Entrenamiento y regeneración del modelo

1. Abre `notebooks/knn_voter_intention.ipynb` en Google Colab.
2. Sube el CSV `voter_intentions_3000.csv` a la sesión de Colab (o monta Drive).
3. Ejecuta todas las celdas. El notebook:
   - Realiza EDA mínimo y gráficas.
   - Prueba varios valores de K y selecciona el mejor.
   - Entrena el pipeline completo (imputación + codificación + escalado + KNN).
   - Guarda el modelo en `models/knn_voter_intention_pipeline.pkl` junto con metadatos de clases y columnas.
4. Descarga `models/knn_voter_intention_pipeline.pkl` y colócalo en esta carpeta `backend/models/`.

> Nota: El archivo de dataset no se versiona. Debe proveerse externamente.
> Para facilitar pruebas locales, el repo incluye un artefacto sintético generado con `app/ml/train_dummy_model.py`. Sustitúyelo por el modelo real entrenado con el CSV.

## Docker

Construir y correr en local:

```bash
cd backend
docker build -t knn-vote-backend .
docker run -p 8000:8000 -e MODEL_PATH=/app/models/knn_voter_intention_pipeline.pkl knn-vote-backend
```

Para Render u otro PaaS:

- Imagen basada en `python:3.11-slim`
- Comando: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Puerto expuesto: `8000`

## Limitaciones

- Modelo educativo; los datos pueden ser sintéticos o incompletos.
- KNN es sensible a escala y desbalance; revisar métricas y re-entrenar con datos actualizados antes de cualquier uso real en campaña.
