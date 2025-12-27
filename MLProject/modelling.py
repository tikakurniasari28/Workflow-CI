import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("n_estimators", type=int)
parser.add_argument("max_depth", type=int)
parser.add_argument("dataset", type=str)
args = parser.parse_args()

if os.getenv("MLFLOW_EXPERIMENT_NAME") is None and os.getenv("MLFLOW_RUN_ID") is None:
    mlflow.set_experiment("CI-Retraining-Experiment")

mlflow.sklearn.autolog()

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / args.dataset
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


os.makedirs("model", exist_ok=True)
mlflow.sklearn.save_model(model, path="model")
