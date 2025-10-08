import subprocess

def run_step(name, cmd):
    print(f"\nEjecutando paso: {name}")
    print(f"   Comando: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error en el paso '{name}'")
    print(f"Paso '{name}' completado correctamente.")

if __name__ == "__main__":
    steps = [
        ("Preprocesamiento", "python src/scripts/preprocess.py --input data/raw/HousingData.csv --params src/mlops/params.yaml --train_out data/curated/HousingDataTrain.csv --test_out data/curated/HousingDataTest.csv"),
        ("Entrenamiento", "python src/scripts/train.py --train data/curated/HousingDataTrain.csv --params src/mlops/params.yaml --model_out models/model.joblib --info_out models/model_info_train.json"),
        ("Evaluación", "python src/scripts/evaluate.py --test data/curated/HousingDataTest.csv --model models/model.joblib --params src/mlops/params.yaml --metrics_out data/monitoring/metrics.json")
    ]

    for name, cmd in steps:
        run_step(name, cmd)

    print("\nPipeline ejecutado correctamente.")



