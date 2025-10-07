# HousingML - Challenge Sr. Machine Learning Eng.

## Descripción del Proyecto

**HousingML** es una solución **MLOps** desarrollada para poner en producción un modelo de *Machine Learning* que predice el **precio de viviendas** del dataset publico open-source [boston-housing-dataset](https://www.kaggle.com/datasets/altavish/boston-housing-dataset).
Incluye un **pipeline reproducible de procesamiento, entrenamiento, y evaluación**, una **API RESTful** para inferencia, reentrenamiento y monitoreo con **FastAPI**, y configuración de **contenedores Docker** para despliegue portátil en cualquier entorno.  

En la arquitectura fueron utilizadas únicamente herramientas **open-source y self-hosted**, no se contempla despliegue de servicios o artefactos en ninguna nube.

---

## Tecnologías Principales

| Componente | Herramienta |
|-------------|--------------|
| Lenguaje base | Python 3.12 |
| manejador de paquetes| uv |
| Framework API | FastAPI |
| Modelado ML | Scikit-learn, XGBoost |
| Serialización | Joblib |
| Orquestación | Scripts Python |
| Contenedorización | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## 📂 Estructura del Proyecto

```
.
├── data/
│   ├── raw/                     # Datos originales (Boston Housing)
│   ├── curated/                 # Datos procesados para entrenamiento
│   └── monitoring/              # Metricas del modelo y dataset de predicciones del servicio
│
├── models/                      # Modelos, scaler de caracteristicas, metricas resultante de entrenamiento del modelo. 
│
├── src/
│   │ 
│   ├── exploration/             # Contiene el Notebook de exploración EDA y Modelado para el dataset propuesto
│   │ 
│   ├── mlops/                   # Incluye el pipeline orquestador de los scripts para mlops
│   │ 
│   ├── repositories/            # Repositorios necesarios para desacoplar servicio de almacenamiento, en este caso almacenamiento local
│   │ 
│   ├── schemas/                 # Esquemas necesarios para estructurar datos, en este caso entradas y salidas de la API
│   │ 
│   ├── scripts/
│   │   ├── preprocess.py        # Limpieza y transformación del dataset
│   │   ├── train.py             # Entrenamiento y persistencia del modelo
│   │   └── evaluate.py          # Evaluación del modelo en conjutos de test
│   │
│   └── main.py                  # API REST FastAPI con endpoints /predict /retrain y /monitor
│
├── tests/                       # Pruebas unitarias con pytest para las funcionalidades expuestas en API
│
├── Dockerfile                   # Imagen base para construir la API 
├── docker-compose.yaml          # Orquestación local y levantar los servicios (API)
├── pyproject.toml               # Definición del proyecto y dependencias principales
├── requirements-dev.txt         # Dependencias adicionales de desarrollo
├── .gitignore                   
├── LICENSE                      
└── README.md                    
```

---

## 🧠 Pipeline de Machine Learning

### 1. Preprocesamiento (`preprocess.py`)
- Carga del dataset **Boston Housing (open source)**.
- Limpieza de datos, eliminación de nulos y outliers.
- Normalización y escalado de variables numéricas.
- División del dataset en `train/test` y almacenamiento en `/data/curated/`.

### 2. Entrenamiento (`train.py`)
- Entrenamiento de modelo **XGBoostRegressor**.
- Evaluación con métricas:
  - `R² Score`
  - `RMSE` (Root Mean Square Error)
- Persistencia del modelo con **joblib** en `models/`.

### 3. Evaluación (`evaluate.py`)
- Carga el modelo entrenado desde `models/model.joblib`.
- Evalúa su desempeño con el dataset de test (`data/curated/HousingDataTest.csv`).
- Calcula y guarda métricas como:
  - `R²`
  - `Adjusted R²`
  - `RMSE`
- Guarda los resultados en `data/monitoring/metrics.json` para monitoreo del modelo.

---

## Estructura de la API

### Endpoint `/predict`
**Método:** `POST`  
**Descripción:** Realiza una predicción de precio usando el modelo en producción. Valida las columnas de entrada, aplica el scaler, devuelve la predicción y guarda el registro (features escaladas + prediction + timestamp) en el CSV de monitoreo.
**Entrada:**
```json
{
    "features": {
      "CRIM": 0.16902,
      "ZN": 0,
      "INDUS": 25.65,
      "CHAS": 0,
      "NOX": 0.581,
      "RM": 5.986,
      "AGE": 88.4,
      "DIS": 1.9929,
      "RAD": 2,
      "TAX": 188,
      "PTRATIO": 19.1,
      "B": 385.02,
      "LSTAT": 14.81
    }
}
```

**Salida:**
```json
{
    "prediction": 21.399450302124023
}
```


## Endpoint `/retrain`
**Método:** `GET`  
**Descripción:** Ejecuta el pipeline completo de **preprocesamiento**, **entrenamiento** y **evaluación**.  
Reentrena el modelo con los datos actuales en raw, actualiza el modelo productivo y devuelve las métricas obtenidas en la evaluación.

**Salida:**
```json
{
  "status": "Modelo reentrenado exitosamente",
  "metrics": {
    "R2": 0.8712,
    "Adjusted_R2": 0.8658,
    "RMSE": 2.6934,
    "model_type": "XGBRegressor"
  }
}
```


### Endpoint `/monitor`
**Método:** `POST`  
**Descripción:** Devuelve el número de predicciones realizadas en los últimos `n` días.  
Lee el archivo de monitoreo (`predictions.csv`) donde se registran las predicciones con su timestamp.

**Entrada:**
```json
{
  "dias": 1
}
```

**Salida:**
```json
{
  "total_predicciones": 24,
  "dias": 1,
  "fecha_inicio": "2025-10-06T14:43:49.926470",
  "fecha_fin": "2025-10-07T14:43:49.926470",
  "message": "Se encontraron 24 predicciones en los últimos 1 días (desde 2025-10-06 hasta 2025-10-07)."
}
```
---


## Pruebas Unitarias

Ubicadas en el directorio `tests/` e implementadas con **pytest** y **FastAPI TestClient**.

Se validaron algunos de los principales comportamientos de la API:

- Respuesta correcta del endpoint raíz `/`.
- Predicción exitosa en `/predict` con datos simulados (mock de modelo y scaler).
- Manejo de error por columnas faltantes en `/predict`.
- Cálculo correcto de predicciones recientes en `/monitor`.
- Respuesta `404` cuando no existe el archivo de monitoreo.
- Manejo de errores internos durante el reentrenamiento en `/retrain`.

Ejecutar con:
```bash
uv run pytest -v tests/api_test.py
```

---

## Despliegue con Docker mediante compose

### Usando `docker-compose`
```bash
docker-compose up --build
```
Permite levantar la API en el puerto 8080.

---

## CI/CD con GitHub Actions

El flujo de integración continua se encuentra definido en `.github/workflows/ci.yml` y se ejecuta automáticamente en cada `push` o `pull request` hacia las ramas `main`, `certification` y `develop`, solo se implementa el CI puesto que para el CD necesitaríamos desplegar el artefacto, paso no contemplado para el requerimiento.

### Flujo del pipeline
1. **Configura el entorno** 
2. **Instala dependencias** usa `uv` para sincronizar los paquetes del proyecto definidos en `pyproject.toml`.  
3. **Ejecuta el pipeline MLOps** corre el script `src/mlops/pipeline.py`, que automatiza las fases de preprocesamiento, entrenamiento y evaluación del modelo.  
4. **Ejecuta las pruebas unitarias automáticas** valida la API con `pytest` ejecutando `tests/api_test.py`, garantizando la funcionalidad de los endpoints.

Este pipeline asegura la **integración continua** del proyecto, validando que el modelo y la API sean funcionales antes de integrar cambios en las ramas principales.

---

## Monitoreo y Reentrenamiento

- El endpoint `/predict` registra automáticamente cada predicción en el archivo `data/monitoring/predictions.csv`, incluyendo las features escaladas, la predicción generada y un timestamp.
- El endpoint `/monitor` permite consultar cuántas predicciones se han realizado en los últimos `n` días, filtrando las fechas dentro del archivo de monitoreo.
- El endpoint `/retrain` ejecuta el pipeline completo de preprocesamiento, entrenamiento y evaluación, actualizando el modelo productivo y sus métricas.
- Todo el monitoreo se realiza mediante archivos CSV y JSON locales, sin dependencias externas ni servicios de terceros.

---

## Dependencias

Definidas en `pyproject.toml` y gestionadas por **uv**:

**Principales:**
- `fastapi[all]==0.118.0`
- `scikit-learn==1.7.2`
- `xgboost==3.0.5`
- `joblib==1.5.2`
- `pandas==2.3.3`
- `pydantic==2.11.10`

**Desarrollo (`requirements-dev.txt`):**
- `matplotlib`, `seaborn`, `statsmodels`, `scipy`, `ipykernel`

Instalación:
```bash
uv sync --frozen
```

---

## Ejecución Local

### Entrenamiento del modelo mlops (desde la raiz del proyecto)
```bash
uv run src/mlops/pipeline.py
```

### Ejecución de la API (desde la carpeta src del proyecto)
```bash
uv run uvicorn main:app --port 8080 --reload
```

### Prueba de inferencia
```bash
curl -X POST http://localhost:8080/predict      -H "Content-Type: application/json"      -d '{"features": {"CRIM": 0.16902, "ZN": 0, "INDUS": 25.65, "CHAS": 0, "NOX": 0.581, "RM": 5.986, "AGE": 88.4, "DIS": 1.9929, "RAD": 2, "TAX": 188, "PTRATIO": 19.1, "B": 385.02, "LSTAT": 14.81}}'
```

---

## Posibles Mejoras

- Implementar MLflow para versionado y monitoreo de métricas y modelo.  
- Agregar reentrenamiento automatizado con scheduler cuando se sepa cada cuanto se actualizará la data.  
- Incorporar pruebas de performance y estrés sobre la API, y aumentar coverage de pruebas sobre demás servicios y funcionalidades del proyecto.  
- Mejorar sistema de logs de infromación en los diferentes procesos del proyecto para trazabilidad.  
- Añadir alertas por drift o degradación de desempeño del modelo, basado en el historico de predicciones.

---

## Uso de Herramientas de IA

El uso realizado de IA dentro del proyecto fue el asistente técnico para soporte en generación de documentación y depuración de código.

---
