from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pathlib import Path
import os
import json
from datetime import datetime, timedelta
import subprocess
import logging
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.schemas.service_schemas import PredictRequest, PredictResponse, RetrainResponse, MonitorResponse, MonitorRequest
from src.repositories.database_repository import DatabaseRepository

from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "../models/model.joblib")
MONITOR_CSV = os.getenv("MONITOR_CSV", "../data/monitoring/predictions.csv")
RAW_CSV = os.getenv("RAW_CSV", "../data/raw/HousingData.csv")
TRAIN_CSV = os.getenv("TRAIN_CSV", "../data/curated/HousingDataTrain.csv")
TEST_CSV = os.getenv("TEST_CSV", "../data/curated/HousingDataTest.csv")
SCALER_PATH = os.getenv("SCALER_PATH", "../models/scaler.joblib")
PARAMS_PATH = os.getenv("PARAMS_PATH", "mlops/params.yaml")
MODEL_INFO_PATH = os.getenv("MODEL_INFO_PATH", "../models/model_info_train.json")
METRICS_PATH = os.getenv("METRICS_PATH", "../data/monitoring/metrics.json")


train_repo = DatabaseRepository(TRAIN_CSV)
monitor_repo = DatabaseRepository(MONITOR_CSV)

model = None
if Path(MODEL_PATH).exists():
    try:
        model = joblib.load(MODEL_PATH)
        logging.info(f"Modelo cargado correctamente desde {MODEL_PATH}")
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        model = None
else:
    logging.warning(f"Modelo no encontrado en {MODEL_PATH}. Ejecute el entrenamiento primero.")
    model = None

scaler = None
if Path(SCALER_PATH).exists():
    try:
        scaler = joblib.load(SCALER_PATH)
        logging.info(f"Scaler cargado correctamente desde {SCALER_PATH}")
    except Exception as e:
        logging.error(f"Error al cargar el scaler: {e}")
else:
    logging.warning(f"Scaler no encontrado en {SCALER_PATH}. Ejecute el entrenamiento primero.")


app = FastAPI(title="HousePricePredictor")

@app.get("/")
def read_root():
    return {"message": "Servicio de predicción de precios de viviendas. Use /docs para ver la documentación."}


@app.post("/predict", response_model=PredictResponse, summary="Realiza una predicción y guarda el input con la predicción y timestamp.")
def predict(req: PredictRequest):
    """
    Realiza la predicción usando el modelo actual.
    Guarda el input original junto con la predicción y el timestamp en el repositorio de monitoreo.
    """
    if model is None:
        raise HTTPException(status_code=400, detail="El modelo no está disponible. Ejecute el entrenamiento primero.")

    if scaler is None:
        raise HTTPException(status_code=400, detail="El scaler no está disponible. Ejecute el entrenamiento primero.")

    try:
        train_df = train_repo.load()
        expected_cols = [col for col in train_df.columns if col != 'MEDV']

        # Validar columnas
        input_features = req.features.copy()
        missing = [col for col in expected_cols if col not in input_features]
        extra = [col for col in input_features if col not in expected_cols]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")
        
        X_pred = pd.DataFrame([input_features])
        if 'RAD' in X_pred.columns:
            X_pred = X_pred.drop(columns='RAD')
        
        X_pred_scaled = scaler.transform(X_pred)
        X_pred_scaled_df = pd.DataFrame(X_pred_scaled, columns=X_pred.columns)
        pred = float(model.predict(X_pred_scaled)[0])
        row = X_pred_scaled_df.iloc[0].to_dict()
        row["prediction"] = pred
        row["timestamp"] = datetime.now().isoformat()
        df = pd.DataFrame([row])
        monitor_repo.append(df, MONITOR_CSV)
        return PredictResponse(prediction=pred)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/retrain", response_model=RetrainResponse, summary="Reentrena el modelo y actualiza el modelo productivo.")
def retrain():
    """
    Reentrena el modelo usando los scripts de procesamiento y entrenamiento.
    Actualiza el modelo productivo y retorna las métricas principales.
    """
    try:
        preprocess_cmd = ["python", "scripts/preprocess.py", "--input", RAW_CSV, "--params", PARAMS_PATH, "--train_out", TRAIN_CSV, "--test_out", TEST_CSV, "--scaler_out", SCALER_PATH]
        subprocess.run(preprocess_cmd, check=True)

        train_cmd = ["python", "scripts/train.py", "--train", TRAIN_CSV, "--params", PARAMS_PATH, "--model_out", MODEL_PATH, "--info_out", MODEL_INFO_PATH]
        subprocess.run(train_cmd, check=True)

        evaluate_cmd = ["python", "scripts/evaluate.py", "--test", TEST_CSV, "--model", MODEL_PATH, "--params", PARAMS_PATH, "--metrics_out", METRICS_PATH]
        subprocess.run(evaluate_cmd, check=True)

        metrics = {}
        if Path(METRICS_PATH).exists():
            with open(METRICS_PATH) as f:
                metrics = json.load(f)

        global model
        model = joblib.load(MODEL_PATH)

        return RetrainResponse(status="Modelo reentrenado exitosamente", metrics=metrics)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error en reentrenamiento: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitor", response_model=MonitorResponse, summary="Cuenta cuántas predicciones se han hecho en los últimos N días.")
def monitor(request: MonitorRequest):
    """
    Cuenta cuántas predicciones se han realizado en los últimos `n` días.
    Usa el archivo de monitoreo donde cada predicción está registrada con una columna 'timestamp'.
    """
    try:
        if not Path(MONITOR_CSV).exists():
            raise HTTPException(status_code=404, detail="No se encontró el archivo de monitoreo. Aún no hay registros.")
        df = monitor_repo.load()
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        fecha_fin = datetime.now()
        fecha_inicio = fecha_fin - timedelta(days=request.dias)

        mask = (df["timestamp"] >= fecha_inicio) & (df["timestamp"] <= fecha_fin)
        df_filtrado = df.loc[mask]
        total_predicciones = len(df_filtrado)

        message = (
            f"Se encontraron {total_predicciones} predicciones en los últimos {request.dias} días "
            f"(desde {fecha_inicio.date()} hasta {fecha_fin.date()})."
        )

        return MonitorResponse(
            total_predicciones=total_predicciones,
            dias=request.dias,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    