
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse
import yaml
import joblib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
#from repositories.database_repository import DatabaseRepository
from repositories.database_repository import DatabaseRepository

class PreprocessingService:
    """
    Servicio para procesar los datos para el modelo ML Housing Prices.
    Realiza limpieza, transformación y split delegando IO al repository.
    """
    def __init__(self, repository: DatabaseRepository, params: dict):
        self.repository = repository
        self.params = params

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina nulos y columna correlacionada RAD."""
        df = df.dropna()
        if 'RAD' in df.columns:
            df = df.drop(columns='RAD')
        return df

    def split_features(self, df: pd.DataFrame):
        """Separa features y target."""
        X = df.drop(columns='MEDV', axis=1)
        y = df['MEDV']
        return X, y

    def scale_features(self, X: pd.DataFrame):
        """Escala features y retorna el scaler."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    def split_data(self, X_scaled: pd.DataFrame, y: pd.Series):
        """Divide en train/test."""
        test_size = self.params.get('preprocessing', {}).get('test_size', 0.3)
        random_state = self.params.get('preprocessing', {}).get('random_state', 42)
        return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    def save_artifacts(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        scaler: StandardScaler,
        train_path: str,
        test_path: str,
        scaler_path: str
    ):
        """Guarda los datasets y el scaler."""
        self.repository.save(train_df, train_path)
        self.repository.save(test_df, test_path)
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    def run(self, train_path: str, test_path: str, scaler_path: str):
        """Orquesta el procesamiento completo."""
        df = self.repository.load()
        df = self.clean_data(df)
        X, y = self.split_features(df)
        X_scaled, scaler = self.scale_features(X)
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y)
        train_df = pd.DataFrame(X_train, columns=X.columns)
        train_df['MEDV'] = y_train.reset_index(drop=True)
        test_df = pd.DataFrame(X_test, columns=X.columns)
        test_df['MEDV'] = y_test.reset_index(drop=True)
        self.save_artifacts(train_df, test_df, scaler, train_path, test_path, scaler_path)


def load_params(params_path):
    """Carga los parámetros de configuración desde YAML."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/HousingData.csv")
    parser.add_argument("--params", default="mlops/params.yaml")
    parser.add_argument("--train_out", default="data/curated/HousingDataTrain.csv")
    parser.add_argument("--test_out", default="data/curated/HousingDataTest.csv")
    parser.add_argument("--scaler_out", default="models/scaler.joblib")
    args = parser.parse_args()

    params = load_params(args.params)
    repo = DatabaseRepository(args.input)
    service = PreprocessingService(repo, params)
    service.run(args.train_out, args.test_out, args.scaler_out)