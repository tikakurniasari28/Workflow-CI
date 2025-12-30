import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_estimators", type=int)
    parser.add_argument("max_depth", type=int)
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    y = df["Confirmed"]
    X = df.drop(columns=["Confirmed"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("mse", mse)

        mlflow.sklearn.log_model(model, "model")

        print("Training selesai, MSE:", mse)

if __name__ == "__main__":
    main()
