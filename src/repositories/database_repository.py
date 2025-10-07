import pandas as pd
from pathlib import Path

class DatabaseRepository:
    """
    Repository para operaciones de acceso a datos en archivos CSV.
    """
    def __init__(self, input_path: str = None):
        """
        Inicializa el repository con un path de entrada opcional.
        Args:
            input_path (str): Ruta al archivo CSV de entrada.
        """
        self.input_path = input_path

    def load(self, path: str = None) -> pd.DataFrame:
        """
        Carga un DataFrame desde un archivo CSV.
        Args:
            path (str): Ruta al archivo CSV. Si no se especifica, usa el path de la instancia.
        Returns:
            pd.DataFrame: DataFrame cargado.
        """
        csv_path = path if path is not None else self.input_path
        if not csv_path:
            raise ValueError("No se especificó la ruta de entrada para cargar datos.")
        return pd.read_csv(csv_path)

    def save(self, df: pd.DataFrame, path: str):
        """
        Guarda un DataFrame en un archivo CSV, creando el directorio si es necesario.
        Args:
            df (pd.DataFrame): DataFrame a guardar.
            path (str): Ruta destino del archivo CSV.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    
    def append(self, df: pd.DataFrame, path: str):
        """
        Agrega nuevas filas a un archivo CSV existente o crea el archivo si no existe.
        Args:
            df (pd.DataFrame): DataFrame con las filas a agregar.
            path (str): Ruta del archivo CSV.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if Path(path).exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)