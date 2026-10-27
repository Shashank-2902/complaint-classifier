"""A small API that classifies one complaint.

    uvicorn api.main:app --reload       then open http://localhost:8000/docs

The model is read from the folder MODEL_DIR (default: deploy/model, made by scripts/export_registered_model.py).
If the model is not sure (confidence below the threshold), the answer says "needs human review".
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "deploy" / "model"))
MODEL_FILE = "tfidf_lr.joblib"
MAX_TEXT_CHARS = 5000

state = {"bundle": None}


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=10, json_schema_extra={"maxLength": MAX_TEXT_CHARS},
                      examples=["The air bag light came on and the steering wheel locked while driving."])
    threshold: float | None = Field(default=None, ge=0, le=1,
                                    description="Confidence below this value means 'needs human review'. Default: the stored threshold.")

    @field_validator("text")
    @classmethod
    def text_must_not_be_too_long(cls, text):
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"the text has {len(text)} characters, the limit is {MAX_TEXT_CHARS}")
        return text


def get_bundle():
    if state["bundle"] is None:
        path = MODEL_DIR / MODEL_FILE
        if not path.exists():
            raise HTTPException(503, f"No model found at {path}. Build it with: python scripts/export_registered_model.py")
        state["bundle"] = joblib.load(path)
    return state["bundle"]


def version_of(bundle):
    return str(bundle.get("model_version", "unknown"))


@asynccontextmanager
async def lifespan(app):
    try:
        get_bundle()        # load the model when the server starts, so the first request is not slow
    except HTTPException:
        pass                # /health and /classify answer 503 until a model is there
    yield


app = FastAPI(title="Complaint classifier", lifespan=lifespan,
              description="Predicts the vehicle component of a complaint text, or asks for human review when unsure.")


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
        "model_version": version_of(bundle),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_version": version_of(get_bundle())}


@app.get("/classes")
def classes():
    return {"classes": get_bundle()["classes"]}


@app.post("/classify")
def classify(request: ClassifyRequest):
    return classify_text(get_bundle(), request.text, request.threshold)
