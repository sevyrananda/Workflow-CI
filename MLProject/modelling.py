import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import argparse
import os

def main(n_estimators, max_depth, dataset):
    mlflow.set_artifact_uri(os.getenv("MLFLOW_ARTIFACT_URI", "mlruns"))
    mlflow.sklearn.autolog()

    # Load data
    data = pd.read_csv(dataset)

    # Konversi kolom integer ke float64
    X = data.drop(columns="TYPE")
    X = X.astype({col: 'float64' for col in X.select_dtypes('int').columns})
    y = data["TYPE"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        signature = infer_signature(X_test, predictions)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:5]
        )

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        print(f"Model logged to MLflow with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=505)
    parser.add_argument("--max_depth", type=int, default=37)
    parser.add_argument("--dataset", type=str, default="seed_preprocessing.csv")
    args = parser.parse_args()

    main(args.n_estimators, args.max_depth, args.dataset)