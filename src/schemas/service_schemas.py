from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

class PredictRequest(BaseModel):
    """
    Cuerpo de la solicitud para el endpoint de predicción. 
    Las características deben coincidir con las columnas utilizadas en el entrenamiento.
    """
    features: Dict[str, float] = Field(..., description="Valores de las características para la predicción.")

class PredictResponse(BaseModel):
    """
    Respuesta del endpoint de predicción.
    """
    prediction: float = Field(..., description="Valor predicho por el modelo.")

class RetrainResponse(BaseModel):
    """
    Respuesta del endpoint de reentrenamiento.
    """
    status: str = Field(..., description="Mensaje con el estado del reentrenamiento.")
    metrics: Dict[str, Any] = Field(..., description="Métricas obtenidas durante el entrenamiento.")

class MonitorRequest(BaseModel):
    dias: int = Field(..., gt=0, description="Número de días hacia atrás para contar las predicciones.")

class MonitorResponse(BaseModel):
    total_predicciones: int
    dias: int
    fecha_inicio: datetime
    fecha_fin: datetime
    message: str
