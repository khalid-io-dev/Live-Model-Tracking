import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Diabetes Risk Demo", layout="centered")

st.title("Diabetes Risk Estimator")
st.markdown(
    """
This demo uses a previously-trained classifier to estimate **diabetes risk** (binary: high/low).
Enter the clinical values on the left and click **Predict**.
"""
)

# -------------------------
# Helpers: load artifacts
# -------------------------
@st.cache_resource
def load_artifact(path_list):
    """Try a list of possible filenames, return first that loads."""
    for p in path_list:
        if p is None:
            continue
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception:
                try:
                    return joblib.load(p, mmap_mode=None)
                except Exception:
                    continue
    return None

# expected artifact file names 
model_files = ["final_model.pkl", "final_model.joblib", "best_model.joblib"]
imputer_files = ["imputer.pkl", "imputer.joblib"]
scaler_files = ["scaler.pkl", "scaler.joblib"]
features_files = ["feature_names.pkl", "feature_names.joblib"]

model = load_artifact(model_files)
imputer = load_artifact(imputer_files)
scaler = load_artifact(scaler_files)
feature_names = load_artifact(features_files)

# small fallback if loaded as bytes or list
if isinstance(feature_names, (bytes, str)):
    try:
        feature_names = joblib.load(feature_names)
    except Exception:
        pass

default_feature_order = ["Glucose", "BMI", "DiabetesPedigreeFunction",
                         "BloodPressure", "SkinThickness", "Insulin", "Age", "Pregnancies"]

if feature_names is None:
    feature_names = [c for c in default_feature_order if True]  # fallback
    st.warning("feature names not found — using default feature list. If predictions fail, ensure 'feature_names.pkl' exists.")

# Build reasonable default values dictionary:
def build_defaults(feature_names, imputer_obj=None):
    defaults = {}

    if imputer_obj is not None and hasattr(imputer_obj, "statistics_"):
        stats = np.array(imputer_obj.statistics_)
        # If feature_names were stored, imputer stats should correspond to same order.
        for i, feat in enumerate(feature_names):
            if i < len(stats):
                defaults[feat] = float(stats[i])
            else:
                defaults[feat] = 0.0
    else:
        # sensible clinical defaults (medians typical for Pima):
        fallback = {
            "Glucose": 117.0,
            "BMI": 32.0,
            "DiabetesPedigreeFunction": 0.37,
            "BloodPressure": 72.0,
            "SkinThickness": 23.0,
            "Insulin": 30.5,
            "Age": 29,
            "Pregnancies": 3
        }
        for feat in feature_names:
            defaults[feat] = float(fallback.get(feat, 0.0))
    return defaults

defaults = build_defaults(feature_names, imputer)

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("Enter clinical features")
user_data = {}
for feat in feature_names:
    # choose input type and ranges sensibly
    if "Age" in feat or "Pregnanc" in feat or "children" in feat.lower():
        val = st.sidebar.number_input(feat, min_value=0, max_value=120, value=int(defaults.get(feat, 0)), step=1)
    else:
        # float input with one decimal by default
        val = st.sidebar.number_input(feat, value=float(defaults.get(feat, 0.0)), format="%.3f")
    user_data[feat] = val

st.sidebar.markdown("---")
use_thresholds = st.sidebar.checkbox("Show clinical threshold flags (Glucose>126, BMI>30, DPF>0.5)", value=True)
st.sidebar.markdown("")

# -------------------------
# Inference button
# -------------------------
if st.button("Predict risk"):
    if model is None:
        st.error("Model artifact not found. Place 'final_model.pkl' (or similar) in the app folder.")
    else:
        # build DataFrame from user inputs in the order expected
        X_user = pd.DataFrame([user_data], columns=feature_names)

        # apply imputer if present 
        if imputer is not None:
            try:
                X_imp = pd.DataFrame(imputer.transform(X_user), columns=X_user.columns, index=[0])
            except Exception as e:
                st.warning(f"Imputer transform failed: {e}. Proceeding without imputation.")
                X_imp = X_user.copy()
        else:
            X_imp = X_user.copy()

        # apply scaler if present
        if scaler is not None:
            try:
                X_scaled_user = pd.DataFrame(scaler.transform(X_imp), columns=X_imp.columns, index=[0])
            except Exception as e:
                st.warning(f"Scaler transform failed: {e}. Proceeding without scaling.")
                X_scaled_user = X_imp.copy()
        else:
            X_scaled_user = X_imp.copy()

        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled_user)[:, 1][0]
            else:
                if hasattr(model, "decision_function"):
                    # scale decision to [0,1] via logistic for display only
                    dfval = model.decision_function(X_scaled_user)[0]
                    proba = 1 / (1 + np.exp(-dfval))
                else:
                    pred_label = model.predict(X_scaled_user)[0]
                    proba = float(pred_label)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            raise

        label = 1 if proba >= 0.5 else 0
        st.metric("Predicted risk (probability of HIGH risk)", f"{proba:.3f}", delta=None)
        st.markdown(f"**Predicted label:** {'HIGH RISK' if label==1 else 'LOW RISK'}")

        if use_thresholds:
            st.subheader("Clinical threshold summary")
            flags = {}
            flags["Glucose > 126"] = float(user_data.get("Glucose", 0)) > 126
            flags["BMI > 30"] = float(user_data.get("BMI", 0)) > 30
            flags["DiabetesPedigreeFunction > 0.5"] = float(user_data.get("DiabetesPedigreeFunction", 0)) > 0.5

            flags_df = pd.DataFrame.from_dict(flags, orient="index", columns=["Exceeded?"])
            flags_df["Exceeded?"] = flags_df["Exceeded?"].map({True: "Yes", False: "No"})
            st.table(flags_df)

            st.markdown(
                "These threshold checks are a simple interpretation aid (spec: Glucose>126, BMI>30, DPF>0.5 indicate high risk). "
                "They do not replace the model prediction but provide clinical context."
            )

        with st.expander("Show input & preprocessed features"):
            st.write("Raw inputs")
            st.write(X_user.T)
            st.write("After imputation")
            st.write(X_imp.T)
            st.write("After scaling (if scaler available)")
            st.write(X_scaled_user.T)

        st.subheader("Feature values (raw)")
        st.bar_chart(X_user.T)

        st.success("Done — interpretation.")
else:
    st.info("Fill inputs on the left and click **Predict** to see the risk estimate.")
