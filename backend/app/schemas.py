from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool