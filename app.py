from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from train_model import MODEL_PATH, train_and_save_outputs

PROJECT_DIR = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_DIR / "index.html"

app = FastAPI(
    title="Research Center Quality Classifier",
    description="Predicts whether a research center belongs to the Premium, Standard, or Basic quality tier.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchCenterInput(BaseModel):
    internalFacilitiesCount: float = Field(..., ge=0, description="Number of internal research facilities.")
    hospitals_10km: float = Field(..., ge=0, description="Hospitals available within 10 km.")
    pharmacies_10km: float = Field(..., ge=0, description="Pharmacies available within 10 km.")
    facilityDiversity_10km: float = Field(
        ...,
        ge=0,
        le=1,
        description="Diversity score for nearby facilities.",
    )
    facilityDensity_10km: float = Field(..., ge=0, description="Density of nearby facilities.")


class PredictionResponse(BaseModel):
    predictedCluster: int
    predictedCategory: str


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    """Load the saved model bundle, training it first if needed."""
    if not MODEL_PATH.exists():
        train_and_save_outputs()
    return joblib.load(MODEL_PATH)


@app.get("/")
def read_root():
    """Serve the single-file frontend when available."""
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    return {
        "message": "Research Center Quality Classifier API",
        "docs": "/docs",
        "predictionEndpoint": "/predict",
    }


@app.get("/health")
def health_check() -> dict[str, object]:
    """Return a simple readiness response and key model metadata."""
    bundle = load_bundle()
    return {
        "status": "ok",
        "modelReady": True,
        "silhouetteScore": bundle["metrics"]["silhouette_score"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_quality(data: ResearchCenterInput) -> PredictionResponse:
    """Predict the quality tier for a new research center."""
    bundle = load_bundle()

    try:
        input_frame = pd.DataFrame([data.model_dump()])
        input_frame = input_frame[bundle["selected_features"]]

        scaled_input = bundle["scaler"].transform(input_frame)
        cluster_label = int(bundle["model"].predict(scaled_input)[0])
        predicted_category = bundle["cluster_to_tier"][cluster_label]

        return PredictionResponse(
            predictedCluster=cluster_label,
            predictedCategory=predicted_category,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
