import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path


def main(n_estimators, max_depth, dataset_path):
    # Set tracking URI ke folder lokal agar bisa diupload ke artifact
    mlruns_path = Path("mlruns").resolve().as_posix()
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")
    mlflow.set_experiment("RandomForest-Experiment")

    # Load dataset
    data = pd.read_csv(dataset_path)
    X = data.drop("TYPE", axis=1)
    y = data["TYPE"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        # Ambil run ID
        run_id = run.info.run_id

        # Simpan run_id ke file
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log params and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained with accuracy: {acc}")
        print(f"Run ID: {run_id} saved to run_id.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="MLProject/seed_preprocessing.csv")
    args = parser.parse_args()

    main(args.n_estimators, args.max_depth, args.dataset)
