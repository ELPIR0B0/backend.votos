from typing import List, Optional

from pydantic import BaseModel, Field


class VoterFeatures(BaseModel):
    # Variables demográficas y socioeconómicas
    age: Optional[float] = Field(
        None, description="Edad en años (por ejemplo 18-99)"
    )
    gender: Optional[str] = Field(None, description="Género reportado en la encuesta")
    education: Optional[str] = Field(None, description="Nivel educativo alcanzado")
    employment_status: Optional[str] = Field(
        None, description="Situación laboral (empleado, desempleado, estudiante, etc.)"
    )
    employment_sector: Optional[str] = Field(
        None, description="Sector laboral si aplica (privado, público, etc.)"
    )
    income_bracket: Optional[str] = Field(
        None, description="Tramo de ingresos según la encuesta"
    )
    marital_status: Optional[str] = Field(None, description="Estado civil")
    household_size: Optional[float] = Field(
        None, description="Número de integrantes del hogar"
    )
    has_children: Optional[str] = Field(
        None, description="Indica si tiene hijos (sí/no/booleano)"
    )
    urbanicity: Optional[str] = Field(
        None, description="Tipo de zona: urbana, suburbana, rural"
    )
    region: Optional[str] = Field(None, description="Región geográfica")
    voted_last: Optional[str] = Field(
        None, description="Si votó en la última elección (sí/no)"
    )
    party_id_strength: Optional[float] = Field(
        None, description="Fuerza de identificación partidista"
    )
    union_member: Optional[str] = Field(
        None, description="Afiliación sindical (sí/no)"
    )
    public_sector: Optional[str] = Field(
        None, description="Trabajo en sector público (sí/no)"
    )
    home_owner: Optional[str] = Field(None, description="Propiedad de vivienda (sí/no)")
    small_biz_owner: Optional[str] = Field(
        None, description="Propietario de pequeño negocio (sí/no)"
    )
    owns_car: Optional[str] = Field(None, description="Propietario de automóvil (sí/no)")
    wa_groups: Optional[str] = Field(
        None,
        description="Pertenencia a grupos de acción/WhatsApp según encuesta original",
    )
    refused_count: Optional[float] = Field(
        None, description="Conteo de preguntas rehusadas"
    )
    attention_check: Optional[float] = Field(
        None, description="Resultado de pregunta de atención"
    )
    will_turnout: Optional[float] = Field(
        None, description="Probabilidad declarada de participar"
    )
    undecided: Optional[float] = Field(
        None, description="Grado en que aún está indeciso"
    )
    preference_strength: Optional[float] = Field(
        None, description="Intensidad de preferencia"
    )
    survey_confidence: Optional[float] = Field(
        None, description="Confianza declarada en la encuesta"
    )
    tv_news_hours: Optional[float] = Field(
        None, description="Horas de consumo de noticias en TV por semana"
    )
    social_media_hours: Optional[float] = Field(
        None, description="Horas de uso de redes sociales por día"
    )
    trust_media: Optional[float] = Field(
        None, description="Nivel de confianza en medios"
    )
    civic_participation: Optional[float] = Field(
        None, description="Participación cívica declarada"
    )
    job_tenure_years: Optional[float] = Field(
        None, description="Años en el empleo actual"
    )
    primary_choice: Optional[str] = Field(
        None, description="Elección primaria declarada"
    )
    secondary_choice: Optional[str] = Field(
        None, description="Elección secundaria declarada"
    )
    intended_vote: Optional[str] = Field(
        None,
        description=(
            "Variable objetivo en el dataset original. No es necesaria para predicción,"
            " pero se permite para evaluar envíos de prueba."
        ),
    )

    class Config:
        anystr_strip_whitespace = True
        extra = "allow"


class PredictResponse(BaseModel):
    predicted_vote: str = Field(..., description="Etiqueta de intención de voto")
    class_probabilities: Optional[list] = Field(
        None, description="Probabilidad por clase si está disponible"
    )
    nearest_neighbors_info: Optional[dict] = Field(
        None, description="Distancias/índices de vecinos más cercanos"
    )


class PredictBatchResponse(BaseModel):
    predictions: List[PredictResponse]


class ModelInfo(BaseModel):
    model_type: str
    k_value: int
    metric: str
    weights: str
    trained_at: Optional[str]
    train_size: Optional[int]
    val_accuracy: Optional[float]
    test_accuracy: Optional[float]
    classes: List[str]
    feature_columns: List[str]
    notes: Optional[str]


class HealthResponse(BaseModel):
    status: str = "ok"
