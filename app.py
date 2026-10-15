"""Streamlit UI for the saved loan-approval classifier."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
REQUESTED_MODEL_PATH = Path(
    os.environ.get("LOAN_MODEL_PATH", ROOT / "loan_prediction_model.pkl")
)
FALLBACK_MODEL_PATH = ROOT / "best_model.pkl"

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered",
)


@st.cache_resource
def load_model(model_path: str) -> Any:
    with Path(model_path).open("rb") as model_file:
        return pickle.load(model_file)


def stop_with_model_error(message: str) -> NoReturn:
    st.error(message)
    st.stop()


model_path = REQUESTED_MODEL_PATH
if not model_path.is_file():
    if REQUESTED_MODEL_PATH != ROOT / "loan_prediction_model.pkl":
        stop_with_model_error(f"Model file not found: {REQUESTED_MODEL_PATH}")
    if FALLBACK_MODEL_PATH.is_file():
        model_path = FALLBACK_MODEL_PATH
        st.info(
            "`loan_prediction_model.pkl` was not found, so this app is using "
            "`best_model.pkl` from the project folder."
        )
    else:
        stop_with_model_error(
            "Model file not found. Put `loan_prediction_model.pkl` in the "
            "project folder, or set the `LOAN_MODEL_PATH` environment variable."
        )

try:
    model = load_model(str(model_path.resolve()))
except (
    OSError,
    pickle.UnpicklingError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    ValueError,
) as error:
    stop_with_model_error(f"Could not load the model at `{model_path}`: {error}")

if not hasattr(model, "feature_names_in_"):
    stop_with_model_error(
        "This model does not contain feature names. Please use the trained "
        "loan-approval pipeline saved from this project."
    )

feature_columns = list(model.feature_names_in_)
applicant_fields = {
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
}
if set(feature_columns) != applicant_fields:
    stop_with_model_error(
        "Model input features do not match this app. "
        f"Expected by model: {feature_columns}"
    )

st.title("Loan Approval Prediction")
st.write(
    "Enter the applicant details, then view the model's estimated result."
)
st.caption(
    "This is an educational project. The prediction is merely an estimate from the model "
    "Not the lender's final decision"
)

with st.form("loan_application"):
    left, right = st.columns(2)
    with left:
        no_of_dependents = st.number_input(
            "Number of dependents",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
        )
        education_label = st.selectbox(
            "Education",
            options=["Graduate", "Not Graduate"],
        )
        self_employed_label = st.selectbox(
            "Self-employed",
            options=["No", "Yes"],
        )
        income_annum = st.number_input(
            "Annual income (dataset units)",
            min_value=0,
            value=5_000_000,
            step=100_000,
        )
        loan_amount = st.number_input(
            "Loan amount (dataset units)",
            min_value=0,
            value=10_000_000,
            step=100_000,
        )
        loan_term = st.number_input(
            "Loan term (dataset units)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )
    with right:
        cibil_score = st.number_input(
            "CIBIL score",
            min_value=300,
            max_value=900,
            value=700,
            step=1,
        )
        residential_assets_value = st.number_input(
            "Residential assets (dataset units)",
            min_value=0,
            value=5_000_000,
            step=100_000,
        )
        commercial_assets_value = st.number_input(
            "Commercial assets (dataset units)",
            min_value=0,
            value=2_000_000,
            step=100_000,
        )
        luxury_assets_value = st.number_input(
            "Luxury assets (dataset units)",
            min_value=0,
            value=5_000_000,
            step=100_000,
        )
        bank_asset_value = st.number_input(
            "Bank assets (dataset units)",
            min_value=0,
            value=1_000_000,
            step=100_000,
        )

    submitted = st.form_submit_button(
        "Predict loan status",
        type="primary",
        use_container_width=True,
    )

if submitted:
    education = {"Graduate": 1, "Not Graduate": 0}[education_label]
    self_employed = {"Yes": 1, "No": 0}[self_employed_label]
    applicant = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }
    input_data = pd.DataFrame(
        [[applicant[column] for column in feature_columns]],
        columns=feature_columns,
    )

    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
    except (ValueError, TypeError, AttributeError) as error:
        st.error(f"Could not make a prediction: {error}")
    else:
        if prediction == 1:
            st.success("Model prediction: Loan Approved")
        else:
            st.error("Model prediction: Loan Rejected")

        classes = list(model.classes_)
        predicted_probability = float(probabilities[classes.index(prediction)])
        st.metric("Model confidence for this result", f"{predicted_probability:.1%}")
