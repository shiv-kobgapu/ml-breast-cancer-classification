from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model" / "saved_models"
TARGET_COLUMN = "target"
CLASS_NAMES = {0: "Malignant", 1: "Benign"}
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])


def calculate_metrics(y_true, y_pred, y_probability):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_probability),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1": f1_score(y_true, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def read_test_data(uploaded_file):
    if uploaded_file is None:
        return pd.read_csv(BASE_DIR / "test_data.csv")
    return pd.read_csv(uploaded_file)


st.set_page_config(page_title="Breast Cancer Classification", page_icon="🔬", layout="wide")

st.title("Breast Cancer Classification")
st.write(
    "Compare five machine learning models using the Breast Cancer Wisconsin "
    "Diagnostic dataset."
)

with st.sidebar:
    st.header("Controls")
    selected_model = st.selectbox("Select a classification model", MODEL_FILES.keys())
    uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type="csv")
    st.caption("If no file is uploaded, the included test_data.csv file is used.")

try:
    data = read_test_data(uploaded_file)
except Exception as error:
    st.error(f"The CSV file could not be read: {error}")
    st.stop()

if TARGET_COLUMN not in data.columns:
    st.error(
        f"The CSV must contain a '{TARGET_COLUMN}' column so that evaluation metrics "
        "can be calculated."
    )
    st.stop()

features = data.drop(columns=[TARGET_COLUMN])
actual_values = data[TARGET_COLUMN]
model = load_model(selected_model)

expected_features = list(model.feature_names_in_)
missing_columns = [column for column in expected_features if column not in features.columns]
extra_columns = [column for column in features.columns if column not in expected_features]

if missing_columns or extra_columns:
    if missing_columns:
        st.error(f"Missing feature columns: {', '.join(missing_columns)}")
    if extra_columns:
        st.error(f"Unexpected feature columns: {', '.join(extra_columns)}")
    st.stop()

features = features[expected_features]

try:
    predicted_values = model.predict(features)
    predicted_probabilities = model.predict_proba(features)[:, 1]
    metrics = calculate_metrics(actual_values, predicted_values, predicted_probabilities)
except Exception as error:
    st.error(f"Model evaluation failed: {error}")
    st.stop()

st.subheader(f"Results: {selected_model}")
metric_columns = st.columns(6)
for column, (metric_name, metric_value) in zip(metric_columns, metrics.items()):
    column.metric(metric_name, f"{metric_value:.4f}")

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Confusion Matrix")
    matrix = confusion_matrix(actual_values, predicted_values, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    st.pyplot(figure)
    plt.close(figure)

with right_column:
    st.subheader("Classification Report")
    report = classification_report(
        actual_values,
        predicted_values,
        labels=[0, 1],
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")

st.subheader("Sample Predictions")
prediction_table = pd.DataFrame(
    {
        "Actual": actual_values.map(CLASS_NAMES),
        "Predicted": pd.Series(predicted_values, index=data.index).map(CLASS_NAMES),
        "Probability of Benign": predicted_probabilities,
    }
)
st.dataframe(prediction_table.head(20), width="stretch")

with st.expander("View uploaded test data"):
    st.dataframe(data, width="stretch")
