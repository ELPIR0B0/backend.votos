import os
from functools import lru_cache


class Settings:
    """
    Configuración simple para la API. Usa variables de entorno cuando estén presentes
    y valores por defecto seguros para desarrollo local.
    """

    def __init__(self) -> None:
        self.model_path: str = os.getenv(
            "MODEL_PATH", os.path.join("models", "knn_voter_intention_pipeline.pkl")
        )
        self.api_title: str = os.getenv(
            "API_TITLE", "Servicio KNN de intención de voto"
        )
        self.api_version: str = os.getenv("API_VERSION", "0.1.0")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"


@lru_cache
def get_settings() -> Settings:
    return Settings()
