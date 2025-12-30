import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

BASE_DIR = Path(__file__).resolve().parent

def main():
    mlflow.set_tracking_uri(f"file://{BASE_DIR}/mlruns")
    mlflow.set_experiment("ci-experiment")

    df = pd.read_csv(BASE_DIR / "day_wise_processed.csv")

    X = df.drop(columns=["Confirmed"])
    y = df["Confirmed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mlflow.log_metric("mse", mean_squared_error(y_test, preds))
    mlflow.log_metric("r2", r2_score(y_test, preds))

    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Training selesai via CI")

if __name__ == "__main__":
    main()
