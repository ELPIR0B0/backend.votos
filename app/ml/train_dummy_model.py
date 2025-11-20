"""
Script de apoyo para generar un artefacto de modelo KNN sintético.
Solo se usa para demos locales cuando no se cuenta con el CSV real.
Para producción, usar el notebook `notebooks/knn_voter_intention.ipynb`.
"""

import random
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Columnas originales del dataset
COLUMNS: List[str] = [
    "age",
    "gender",
    "education",
    "employment_status",
    "employment_sector",
    "income_bracket",
    "marital_status",
    "household_size",
    "has_children",
    "urbanicity",
    "region",
    "voted_last",
    "party_id_strength",
    "union_member",
    "public_sector",
    "home_owner",
    "small_biz_owner",
    "owns_car",
    "wa_groups",
    "refused_count",
    "attention_check",
    "will_turnout",
    "undecided",
    "preference_strength",
    "survey_confidence",
    "tv_news_hours",
    "social_media_hours",
    "trust_media",
    "civic_participation",
    "job_tenure_years",
    "primary_choice",
    "secondary_choice",
    "intended_vote",
]

TARGET = "intended_vote"

CANDIDATES = [
    "CAND_Azon",
    "CAND_Boreal",
    "CAND_Cygnus",
    "CAND_Delta",
    "CAND_Euler",
    "CAND_Fjord",
    "CAND_Gaia",
    "CAND_Hera",
    "CAND_Iota",
    "CAND_Jade",
    "Undecided",
]

CATEGORICAL = [
    "gender",
    "education",
    "employment_status",
    "employment_sector",
    "income_bracket",
    "marital_status",
    "urbanicity",
    "region",
    "voted_last",
    "has_children",
    "union_member",
    "public_sector",
    "home_owner",
    "small_biz_owner",
    "owns_car",
    "wa_groups",
    "primary_choice",
    "secondary_choice",
]


def generate_synthetic_dataframe(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    genders = ["M", "F", "NB"]
    educations = ["HS", "College", "Graduate"]
    employment_statuses = ["Employed", "Unemployed", "Student", "Retired"]
    sectors = ["Private", "Public", "Self", "None"]
    incomes = ["Low", "Mid", "High"]
    marital = ["Single", "Married", "Divorced"]
    urbanicity = ["Urban", "Suburban", "Rural"]
    regions = ["North", "South", "East", "West", "Center"]
    yes_no = ["Yes", "No"]
    wa_groups = ["None", "Community", "Union", "Neighborhood"]
    choices = ["C1", "C2", "C3", "C4"]

    data = {
        "age": rng.integers(18, 80, size=rows),
        "gender": rng.choice(genders, size=rows),
        "education": rng.choice(educations, size=rows),
        "employment_status": rng.choice(employment_statuses, size=rows),
        "employment_sector": rng.choice(sectors, size=rows),
        "income_bracket": rng.choice(incomes, size=rows),
        "marital_status": rng.choice(marital, size=rows),
        "household_size": rng.integers(1, 6, size=rows),
        "has_children": rng.choice(yes_no, size=rows),
        "urbanicity": rng.choice(urbanicity, size=rows),
        "region": rng.choice(regions, size=rows),
        "voted_last": rng.choice(yes_no, size=rows),
        "party_id_strength": rng.integers(0, 10, size=rows),
        "union_member": rng.choice(yes_no, size=rows),
        "public_sector": rng.choice(yes_no, size=rows),
        "home_owner": rng.choice(yes_no, size=rows),
        "small_biz_owner": rng.choice(yes_no, size=rows),
        "owns_car": rng.choice(yes_no, size=rows),
        "wa_groups": rng.choice(wa_groups, size=rows),
        "refused_count": rng.integers(0, 5, size=rows),
        "attention_check": rng.integers(0, 1 + 1, size=rows),
        "will_turnout": rng.uniform(0, 1, size=rows),
        "undecided": rng.uniform(0, 1, size=rows),
        "preference_strength": rng.integers(0, 10, size=rows),
        "survey_confidence": rng.uniform(0, 1, size=rows),
        "tv_news_hours": rng.uniform(0, 20, size=rows),
        "social_media_hours": rng.uniform(0, 10, size=rows),
        "trust_media": rng.uniform(0, 1, size=rows),
        "civic_participation": rng.integers(0, 5, size=rows),
        "job_tenure_years": rng.uniform(0, 30, size=rows),
        "primary_choice": rng.choice(choices, size=rows),
        "secondary_choice": rng.choice(choices, size=rows),
        "intended_vote": rng.choice(CANDIDATES, size=rows, p=_random_probs()),
    }
    return pd.DataFrame(data)


def _random_probs() -> List[float]:
    """Genera una distribución pseudoaleatoria para las clases."""
    weights = np.array([random.random() for _ in range(len(CANDIDATES))])
    weights = weights / weights.sum()
    return weights.tolist()


def train_and_serialize(df: pd.DataFrame, output_path: Path) -> None:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = [col for col in X.columns if col not in CATEGORICAL]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, CATEGORICAL),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)

    clf = KNeighborsClassifier(n_neighbors=7, weights="distance")
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", clf),
        ]
    )
    model.fit(X_train, y_train_enc)
    val_acc = model.score(X_val, y_val_enc)

    metadata = {
        "model_type": "KNeighborsClassifier",
        "k_value": clf.n_neighbors,
        "metric": clf.metric,
        "weights": clf.weights,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "train_size": len(X_train),
        "val_accuracy": float(val_acc),
        "test_accuracy": None,
        "feature_columns": list(X.columns),
        "classes": encoder.classes_.tolist(),
        "notes": (
            "Modelo sintético generado para desarrollo local sin CSV real. "
            "Reemplázalo entrenando el notebook con los datos verdaderos."
        ),
    }

    bundle = {
        "pipeline": model,
        "label_encoder_classes": encoder.classes_.tolist(),
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"Modelo dummy guardado en {output_path} con val_acc={val_acc:.3f}")


if __name__ == "__main__":
    df_syn = generate_synthetic_dataframe(rows=500)
    output = Path(__file__).resolve().parent.parent.parent / "models" / "knn_voter_intention_pipeline.pkl"
    train_and_serialize(df_syn, output)
