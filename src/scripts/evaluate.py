
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
from repositories.database_repository import DatabaseRepository



class EvaluationService:
    """
    Servicio para evaluar el modelo de regresión Housing Prices y guardar métricas de test.
    """
    def __init__(self, repository: DatabaseRepository, model_path: str):
        self.repository = repository
        self.model_path = model_path
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el modelo en la ruta: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo: {e}")

    def evaluate_and_save(self, test_path: str, metrics_out: str):
        """
        Evalúa el modelo sobre el set de test y guarda las métricas principales.
        """
        df = self.repository.load(test_path)
        X = df.drop(columns='MEDV', axis=1)
        y = df['MEDV']
        y_pred = self.model.predict(X)
        # Métricas
        r2 = r2_score(y, y_pred)
        n = X.shape[0]
        p = X.shape[1]
        adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
        RMSE = np.sqrt(mean_squared_error(y, y_pred))
        metrics = {
            "R2": round(r2, 4),
            "Adjusted_R2": round(adjusted_r2, 4),
            "RMSE": round(RMSE, 4),
            "model_type": type(self.model).__name__
        }
        # Guardar métricas
        Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Métricas de test guardadas en {metrics_out}")
        print(metrics)

def load_params(params_path: str) -> dict:
    """Carga los parámetros de configuración desde YAML."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/curated/HousingDataTest.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--params", default="src/mlops/params.yaml")
    parser.add_argument("--metrics_out", default="data/monitoring/metrics.json")
    args = parser.parse_args()

    params = load_params(args.params)
    repository = DatabaseRepository()
    service = EvaluationService(repository, args.model)
    service.evaluate_and_save(args.test, args.metrics_out)