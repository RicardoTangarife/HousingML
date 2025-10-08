import sys
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
from datetime import datetime, timedelta
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.main import app  

client = TestClient(app)


# -----------------------------------------------------------------------------------
# TEST 1: Verifica que la raíz responda correctamente
# -----------------------------------------------------------------------------------
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Servicio de predicción" in response.json()["message"]


# -----------------------------------------------------------------------------------
# TEST 2: Prueba del endpoint /predict con mocks
# -----------------------------------------------------------------------------------
@patch("src.main.model")
@patch("src.main.scaler")
@patch("src.main.train_repo")
@patch("src.main.monitor_repo")
def test_predict_success(mock_monitor_repo, mock_train_repo, mock_scaler, mock_model):
    # Simular DataFrame de entrenamiento
    mock_train_repo.load.return_value = pd.DataFrame(columns=["CRIM", "ZN", "INDUS", "MEDV"])

    # Simular scaler y modelo
    mock_scaler.transform.return_value = [[0.1, 0.2, 0.3]]
    mock_model.predict.return_value = [25.5]
    mock_monitor_repo.append.return_value = None

    body = {
        "features": {"CRIM": 0.2, "ZN": 18.0, "INDUS": 2.3}
    }

    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)


# -----------------------------------------------------------------------------------
# TEST 3: Error si faltan columnas en /predict
# -----------------------------------------------------------------------------------
@patch("src.main.model")
@patch("src.main.scaler")
@patch("src.main.train_repo")
def test_predict_missing_columns(mock_train_repo, mock_scaler, mock_model):
    mock_train_repo.load.return_value = pd.DataFrame(columns=["CRIM", "ZN", "INDUS", "MEDV"])
    body = {"features": {"CRIM": 0.2, "ZN": 18.0}}  # Falta INDUS

    response = client.post("/predict", json=body)
    assert response.status_code == 400
    assert "Faltan columnas" in response.json()["detail"]


# -----------------------------------------------------------------------------------
# TEST 4: Monitoreo de predicciones (caso con registros recientes)
# -----------------------------------------------------------------------------------
@patch("src.main.Path.exists", return_value=True)
@patch("src.main.monitor_repo")
def test_monitor_with_records(mock_monitor_repo, tmp_path):
    now = datetime.now()
    data = pd.DataFrame({
        "timestamp": [
            (now - timedelta(days=1)).isoformat(),
            (now - timedelta(days=4)).isoformat(),
        ],
        "prediction": [10.5, 20.7]
    })
    mock_monitor_repo.load.return_value = data

    body = {"dias": 2}
    response = client.post("/monitor", json=body)
    assert response.status_code == 200
    result = response.json()
    assert "total_predicciones" in result
    assert result["dias"] == 2
    assert "Se encontraron" in result["message"]


# -----------------------------------------------------------------------------------
# TEST 5: Error si no existe el archivo de monitoreo
# -----------------------------------------------------------------------------------
@patch("src.main.Path.exists", return_value=False)
def test_monitor_no_file(mock_exists):
    body = {"dias": 7}
    response = client.post("/monitor", json=body)
    assert response.status_code == 404
    assert "No se encontró el archivo" in response.json()["detail"]


# -----------------------------------------------------------------------------------
# TEST 6: Simulación de error en /retrain
# -----------------------------------------------------------------------------------
@patch("src.main.subprocess.run", side_effect=Exception("Error simulado"))
def test_retrain_error(mock_subprocess):
    response = client.get("/retrain")
    assert response.status_code == 500
    assert "Error simulado" in response.json()["detail"]
