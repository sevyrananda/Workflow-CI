import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Untuk logging ke DagsHub
from dagshub import dagshub_logger
import dagshub
dagshub.init(repo_owner='sevyrananda', repo_name='iris-classification', mlflow=True)

# Load data
df = pd.read_csv('MLProject/iris_preprocessing.csv')
X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow
mlflow.set_experiment("iris_model")

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # Log model
    signature = infer_signature(X_test, preds)
    input_example = X_test.iloc[:1]
    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

    # === Artefak 1: Confusion Matrix ===
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    os.makedirs("plots", exist_ok=True)
    cm_path = "plots/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # === Artefak 2: Feature Importance ===
    feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.title("Feature Importance")
    plt.xlabel("Score")
    plt.ylabel("Features")

    fi_path = "plots/feature_importance.png"
    plt.savefig(fi_path)
    plt.close()

    mlflow.log_artifact(fi_path)