from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.ml.model import ModelService
from app.schemas.prediction import (
    HealthResponse,
    ModelInfo,
    PredictBatchResponse,
    PredictResponse,
    VoterFeatures,
)

settings = get_settings()
app = FastAPI(title=settings.api_title, version=settings.api_version)

# Permitir CORS amplio para facilitar demos y despliegue en Vercel/Render.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model_service = ModelService(settings.model_path)
except Exception as exc:  # pragma: no cover - advertencia en carga inicial
    # La app seguirá levantando, pero endpoints de predicción responderán con error clarificado.
    model_service = None
    load_error = exc
else:
    load_error = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    if model_service is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible: {load_error}",
        )
    info = model_service.get_info()
    return ModelInfo(**info)


@app.post("/predict", response_model=PredictResponse)
def predict(features: VoterFeatures) -> PredictResponse:
    if model_service is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible: {load_error}",
        )
    try:
        result = model_service.predict_single(features.dict())
        return PredictResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar la solicitud: {exc}",
        ) from exc


@app.post("/predict-batch", response_model=PredictBatchResponse)
def predict_batch(items: List[VoterFeatures]) -> PredictBatchResponse:
    if model_service is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible: {load_error}",
        )
    if not items:
        raise HTTPException(status_code=400, detail="La lista de votantes está vacía.")
    try:
        payloads = [item.dict() for item in items]
        raw_predictions = model_service.predict_batch(payloads)
        predictions = [PredictResponse(**pred) for pred in raw_predictions]
        return PredictBatchResponse(predictions=predictions)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar la solicitud: {exc}",
        ) from exc
