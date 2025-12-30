import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

BASE_DIR = Path(__file__).resolve().parent

def main():
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("default")

    dataset = "day_wise_processed.csv"
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

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    print("Training selesai")
    print("MSE:", mse)
    print("R2:", r2)

if __name__ == "__main__":
    main()
