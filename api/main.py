"""A small API that classifies one complaint.

    uvicorn api.main:app --reload       then open http://localhost:8000/docs

If the model is not sure (confidence below the threshold), the answer says "needs human review".
"""
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent.parent / "models" / "tfidf_lr.joblib"))

app = FastAPI(title="Complaint classifier",
              description="Predicts the vehicle component of a complaint text, or asks for human review when unsure.")
state = {"bundle": None}


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=10, max_length=5000, examples=["The air bag light came on and the steering wheel locked while driving."])
    threshold: float | None = Field(default=None, ge=0, le=1,
                                    description="Confidence below this value means 'needs human review'. Default: the stored threshold.")


def get_bundle():
    if state["bundle"] is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"No model found at {MODEL_PATH}. Build it with: python -m src.baselines.export_model")
        state["bundle"] = joblib.load(MODEL_PATH)
    return state["bundle"]


def classify_text(bundle, text, threshold=None, top=3):
    """Return the prediction for one text. Pure function, used by the endpoint and by the tests."""
    probabilities = bundle["classifier"].predict_proba(bundle["vectorizer"].transform([text]))[0]
    order = np.argsort(probabilities)[::-1]
    classes = bundle["classifier"].classes_
    confidence = float(probabilities[order[0]])
    limit = threshold if threshold is not None else bundle["thresholds"][0]
    return {
        "label": str(classes[order[0]]),
        "confidence": round(confidence, 4),
        "needs_human_review": confidence < limit,
        "threshold_used": round(float(limit), 4),
        "top": [{"label": str(classes[i]), "confidence": round(float(probabilities[i]), 4)} for i in order[:top]],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["bundle"] is not None, "model_path": str(MODEL_PATH)}


@app.get("/classes")
def classes():
    return {"classes": get_bundle()["classes"]}


@app.post("/classify")
def classify(request: ClassifyRequest):
    return classify_text(get_bundle(), request.text, request.threshold)
