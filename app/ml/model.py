import json
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class ModelService:
    """
    Servicio de carga y predicción del pipeline KNN serializado.
    El artefacto esperado es un diccionario con:
        - pipeline: sklearn.Pipeline completa (preproceso + clasificador)
        - label_encoder_classes: lista de clases en orden interno
        - metadata: información auxiliar para el endpoint /model-info
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No se encontró el artefacto del modelo en {self.model_path}. "
                "Ejecuta el notebook de entrenamiento para generarlo."
            )
        bundle = joblib.load(self.model_path)
        self.pipeline = bundle.get("pipeline")
        self.classes_ = bundle.get("label_encoder_classes", [])
        self.metadata = bundle.get("metadata", {})
        self.feature_columns = self.metadata.get("feature_columns", [])
        if self.pipeline is None or not self.feature_columns:
            raise ValueError(
                "El artefacto cargado no contiene pipeline o feature_columns válidos."
            )

    def _build_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        # Garantizar presencia y orden de columnas esperadas por el pipeline.
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_columns]
        return df

    def predict_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        df = self._build_dataframe([payload])
        return self._predict(df)[0]

    def predict_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        df = self._build_dataframe(payloads)
        return self._predict(df)

    def _predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if self.pipeline is None:
            raise RuntimeError("Pipeline no cargada.")

        predictions = self.pipeline.predict(df)
        probability_matrix = (
            self.pipeline.predict_proba(df)
            if hasattr(self.pipeline, "predict_proba")
            else None
        )

        neighbors_info = self._nearest_neighbors(df)
        results: List[Dict[str, Any]] = []
        for idx, pred_idx in enumerate(predictions):
            label = self.classes_[pred_idx] if self.classes_ else str(pred_idx)
            class_probs = None
            if probability_matrix is not None and len(self.classes_) == len(
                probability_matrix[idx]
            ):
                class_probs = [
                    {"label": cls, "probability": float(prob)}
                    for cls, prob in zip(self.classes_, probability_matrix[idx])
                ]

            results.append(
                {
                    "predicted_vote": label,
                    "class_probabilities": class_probs,
                    "nearest_neighbors_info": neighbors_info[idx],
                }
            )
        return results

    def _nearest_neighbors(self, df: pd.DataFrame) -> List[Optional[Dict[str, Any]]]:
        """
        Extrae vecinos más cercanos usando el KNN ya entrenado. Si no está disponible,
        devuelve None para cada fila.
        """
        if self.pipeline is None:
            return [None] * len(df)
        try:
            preprocessor = self.pipeline[:-1]
            classifier = self.pipeline.named_steps.get("classifier")
            if classifier is None:
                return [None] * len(df)
            transformed = preprocessor.transform(df)
            distances, indices = classifier.kneighbors(transformed, n_neighbors=3)
            info = []
            for dist_row, idx_row in zip(distances, indices):
                info.append(
                    {
                        "indices": idx_row.tolist(),
                        "distances": [float(d) for d in dist_row],
                    }
                )
            return info
        except Exception:
            return [None] * len(df)

    def get_info(self) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Pipeline no cargada.")
        metadata = self.metadata.copy()
        # Asegurar campos mínimos para /model-info
        metadata.setdefault("model_type", "KNeighborsClassifier")
        metadata.setdefault(
            "k_value", getattr(self.pipeline.named_steps.get("classifier"), "n_neighbors", None)
        )
        metadata.setdefault(
            "metric", getattr(self.pipeline.named_steps.get("classifier"), "metric", None)
        )
        metadata.setdefault(
            "weights", getattr(self.pipeline.named_steps.get("classifier"), "weights", None)
        )
        metadata.setdefault("feature_columns", self.feature_columns)
        metadata.setdefault("classes", self.classes_)
        return metadata
