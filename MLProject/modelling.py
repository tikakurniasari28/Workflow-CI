import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

if os.getenv("MLFLOW_TRACKING_URI") is None:
    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")

if os.getenv("MLFLOW_RUN_ID") is None:
    mlflow.set_experiment("CI-Retraining-Experiment")

mlflow.sklearn.autolog()

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "day_wise_processed.csv"

df = pd.read_csv(data_path)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2:", r2)

mlflow.sklearn.log_model(model, artifact_path="model")
