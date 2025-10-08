
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
from repositories.database_repository import DatabaseRepository



class TrainingService:
    """
    Servicio para entrenar el modelo de regresion Housing Prices y guardar artefactos y métricas.
    """
    def __init__(self, repository: DatabaseRepository, params: dict):
        self.repository = repository
        self.params = params

    def get_model(self):
        model_type = self.params.get('training', {}).get('model_type', 'xgboost.XGBRegressor')
        if model_type == 'xgboost.XGBRegressor':
            p = self.params['training']
            return XGBRegressor(
                n_estimators=p.get('n_estimators', 1000),
                max_depth=p.get('max_depth', 7),
                eta=p.get('eta', 0.1),
                subsample=p.get('subsample', 0.8),
                colsample_bytree=p.get('colsample_bytree', 0.8)
            )
        else:
            raise ValueError(f"Modelo no soportado: {model_type}")

    def train_and_save(self, model_out: str, info_out: str):
        """
        Entrena el modelo y guarda el artefacto y las métricas principales.
        """
        df = self.repository.load()
        X = df.drop(columns='MEDV', axis=1)
        y = df['MEDV']
        model = self.get_model()
        model.fit(X, y)
        y_pred = model.predict(X)
        # Métricas
        cv_score = cross_val_score(estimator = model, X = X, y = y, cv = 10)
        r2 = model.score(X, y)
        n = X.shape[0]
        p = X.shape[1]
        adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
        RMSE = np.sqrt(mean_squared_error(y, y_pred))
        R2 = model.score(X, y)
        CV_R2 = cv_score.mean()
        metrics = {
            "R2": round(R2, 4),
            "Adjusted_R2": round(adjusted_r2, 4),
            "CrossValidated_R2": round(CV_R2, 4),
            "RMSE": round(RMSE, 4),
            "model_type": type(model).__name__
        }
        # Guardar modelo y métricas
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out)
        Path(info_out).parent.mkdir(parents=True, exist_ok=True)
        with open(info_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Modelo guardado en {model_out}")
        print(f"Métricas guardadas en {info_out}")
        print(metrics)

def load_params(params_path: str) -> dict:
    """Carga los parámetros de configuración desde YAML."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/curated/HousingDataTrain.csv")
    parser.add_argument("--params", default="src/mlops/params.yaml")
    parser.add_argument("--model_out", default="models/model.joblib")
    parser.add_argument("--info_out", default="models/model_info_train.json")
    args = parser.parse_args()

    params = load_params(args.params)
    repository = DatabaseRepository(input_path=args.train)
    service = TrainingService(repository, params)
    service.train_and_save(args.model_out, args.info_out)
