from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import predict_for_project
import traceback

app = FastAPI(
    title="GridSight Predictor API",
    version="1.0.0",
    description="GridSight Phase Aware Risk Prediction API"
)


class PredictionRequest(BaseModel):
    data: dict


@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "GridSight Predictor API",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "UP"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        prediction = predict_for_project(request.data)

        return {
            "success": True,
            "prediction": prediction
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
