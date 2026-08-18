from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
RANDOM_STATE = 42


def load_and_split_data():
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target
    return train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [("scaler", StandardScaler()), ("classifier", GaussianNB())]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def calculate_metrics(y_true, y_pred, y_probability):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_probability),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1": f1_score(y_true, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    x_train, x_test, y_train, y_test = load_and_split_data()
    models = build_models()
    file_names = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Gaussian Naive Bayes": "naive_bayes.joblib",
        "Random Forest": "random_forest.joblib",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        results.append(
            {"ML Model Name": model_name, **calculate_metrics(y_test, predictions, probabilities)}
        )
        joblib.dump(model, MODEL_DIR / file_names[model_name])

    test_data = x_test.copy()
    test_data["target"] = y_test
    test_data.to_csv(PROJECT_DIR / "test_data.csv", index=False)

    metrics_table = pd.DataFrame(results)
    metrics_table.to_csv(PROJECT_DIR / "model_metrics.csv", index=False)
    print(metrics_table.round(4).to_string(index=False))
    print(f"\nSaved {len(test_data)} test rows to test_data.csv")


if __name__ == "__main__":
    main()
