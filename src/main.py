from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pathlib import Path
import os
import json
from datetime import datetime
import subprocess
import numpy as np
import logging
from scipy.stats import ks_2samp

from schemas.service_schemas import PredictRequest, PredictResponse, RetrainResponse, MonitorResponse
from repositories.database_repository import DatabaseRepository

from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "../models/model.joblib")
MONITOR_CSV = os.getenv("MONITOR_CSV", "../data/monitoring/predictions.csv")
TRAIN_CSV = os.getenv("TRAIN_CSV", "../data/curated/HousingDataTrain.csv")
TEST_CSV = os.getenv("TEST_CSV", "../data/curated/HousingDataTest.csv")
SCALER_PATH = os.getenv("SCALER_PATH", "../models/scaler.joblib")
PARAMS_PATH = os.getenv("PARAMS_PATH", "../mlops/params.yaml")
MODEL_INFO_PATH = os.getenv("MODEL_INFO_PATH", "../models/model_info_train.json")


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
        preprocess_cmd = ["python", "scripts/preprocess.py", "--input", "../data/raw/HousingData.csv", "--params", PARAMS_PATH, "--train_out", TRAIN_CSV, "--test_out", TEST_CSV, "--scaler_out", SCALER_PATH]
        subprocess.run(preprocess_cmd, check=True)

        train_cmd = ["python", "scripts/train.py", "--train", TRAIN_CSV, "--params", PARAMS_PATH, "--model_out", MODEL_PATH, "--info_out", MODEL_INFO_PATH]
        subprocess.run(train_cmd, check=True)

        metrics = {}
        if Path(MODEL_INFO_PATH).exists():
            with open(MODEL_INFO_PATH) as f:
                metrics = json.load(f)

        global model
        model = joblib.load(MODEL_PATH)

        return RetrainResponse(status="Modelo reentrenado exitosamente", metrics=metrics)

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error en reentrenamiento: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def calculate_psi(expected, actual, bins=10):
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones.
    """
    expected = np.array(expected)
    actual = np.array(actual)
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)
    psi = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-6) / (actual_perc + 1e-6)))
    return float(round(psi, 4))


def calculate_ks(expected, actual):
    """
    Calcula el estadístico KS para detectar drift en variables numéricas.
    """
    stat, p_value = ks_2samp(expected, actual)
    return {"ks_stat": round(stat, 4), "p_value": round(p_value, 4)}


@app.get("/monitor", response_model=MonitorResponse, summary="Evalúa el drift de variables y calcula PSI entre entrenamiento y predicciones recientes.")
def monitor():
    """
    Evalúa el drift de las variables comparando el set de entrenamiento con las predicciones recientes.
    Retorna el PSI por variable y alerta si alguna variable supera el umbral de drift.
    """
    try:
        if not Path(MONITOR_CSV).exists():
            return MonitorResponse(
                psi={},
                drift_alerts=[],
                psi_threshold=0.2,
                message="No se encontró el archivo de monitoreo. Aún no hay datos registrados para comparar."
            )

        train_df = train_repo.load()
        monitor_df = monitor_repo.load()

        if train_df.empty:
            return MonitorResponse(
                psi={},
                drift_alerts=[],
                psi_threshold=0.2,
                message="El conjunto de entrenamiento está vacío. No es posible realizar el monitoreo."
            )

        if monitor_df.empty or len(monitor_df) < 5:
            return MonitorResponse(
                psi={},
                drift_alerts=[],
                psi_threshold=0.2,
                message="El archivo de monitoreo está vacío o tiene pocos registros. No hay datos suficientes para evaluar drift."
            )

        feature_cols = [col for col in train_df.columns if col != 'MEDV' and col in monitor_df.columns]
        psi_results = {}
        ks_results = {}
        drift_alerts = []

        for col in feature_cols:
            psi = calculate_psi(train_df[col], monitor_df[col])
            psi_results[col] = psi
            ks_result = calculate_ks(train_df[col], monitor_df[col])
            ks_results[col] = ks_result
            #if psi > 0.2:
                #drift_alerts.append(col)
            if ks_result["p_value"] < 0.05:
                drift_alerts.append(col)

        message = "Variables con posible drift: " + ", ".join(drift_alerts) if drift_alerts else "Sin drift significativo"

        return MonitorResponse(
            psi=psi_results,
            drift_alerts=drift_alerts,
            psi_threshold=0.2,
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
