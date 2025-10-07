from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

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

class MonitorResponse(BaseModel):
    """
    Respuesta del endpoint de monitoreo.
    """
    psi: Dict[str, float] = Field(..., description="Valores de PSI por variable.")
    drift_alerts: List[str] = Field(..., description="Variables con deriva superior al umbral definido.")
    psi_threshold: float = Field(..., description="Umbral de PSI para generar alerta de deriva.")
    message: str = Field(..., description="Mensaje resumen del estado de deriva del modelo.")
