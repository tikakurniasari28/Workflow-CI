import os
import sys
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parent

def main():
    mlflow.set_tracking_uri("file:./mlruns")

    dataset = "day_wise_processed.csv"
    if len(sys.argv) >= 4:
        dataset = sys.argv[3]

    data_path = BASE_DIR / dataset
    df = pd.read_csv(data_path)

    y = df["Confirmed"]
    X = df.drop(columns=["Confirmed"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    run_id = os.environ.get("MLFLOW_RUN_ID")
    if run_id is None:
        raise ValueError("MLFLOW_RUN_ID tidak ditemukan. Pastikan script dijalankan via mlflow run .")

    client = MlflowClient()

    client.log_metric(run_id, "mse", mse)
    client.log_metric(run_id, "r2", r2)

    mlflow.sklearn.log_model(model, artifact_path="model")

    shutil.rmtree(BASE_DIR / "model", ignore_errors=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=mlflow.get_artifact_uri("model"),
        dst_path=str(BASE_DIR / "model")
    )

    print("Training selesai")
    print("MSE:", mse)
    print("R2:", r2)
    print("Run ID:", run_id)
    print("Model exported to:", BASE_DIR / "model")

if __name__ == "__main__":
    main()
