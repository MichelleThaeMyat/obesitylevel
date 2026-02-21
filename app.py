import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Obesity Level Prediction",
    page_icon=None,
    layout="wide",
)

# Hide "Press Enter to submit form" helper text
st.markdown("""
<style>
    div[data-testid="InputInstructions"] {
        display: none;
    }
    .stNumberInput input:focus {
        border-color: inherit !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Obesity Level Prediction")
st.caption("Educational demo. Not medical advice.")


# ----------------------------
# Load artifacts
# ----------------------------
MODEL_FILE = Path("obesity_best_model.joblib")
SCHEMA_FILE = Path("schema.json")

if not MODEL_FILE.exists() or not SCHEMA_FILE.exists():
    st.error(
        "Missing required files.\n\n"
        "Please place these files in the same folder as this app:\n"
        "- obesity_best_model.joblib\n"
        "- schema.json"
    )
    st.stop()


@st.cache_resource
def load_model(path: str):
    """Load and cache the trained model pipeline."""
    return joblib.load(path)


@st.cache_data
def load_schema(path: str):
    """Load schema.json (feature ranges/options)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model(str(MODEL_FILE))
schema = load_schema(str(SCHEMA_FILE))


# ----------------------------
# Helpers
# ----------------------------
# Force specific features to be integer in the UI + before prediction
INT_FEATURE_HINTS = {
    "Age",  # Age should be integer
    # Add more here only if your dataset truly stores them as integers
    # "NCP",
}

def pretty_label(name: str) -> str:
    """Convert raw feature names into readable labels."""
    replacements = {
        "family_history_with_overweight": "Family history with overweight",
        "FAVC": "High-calorie food consumption (FAVC)",
        "FCVC": "Vegetable consumption frequency (FCVC)",
        "NCP": "Number of main meals per day (NCP)",
        "CAEC": "Eating between meals (CAEC)",
        "SMOKE": "Smoker (SMOKE)",
        "CH2O": "Daily water intake (CH2O)",
        "SCC": "Calorie monitoring (SCC)",
        "FAF": "Physical activity frequency (FAF)",
        "TUE": "Time using technology devices (TUE)",
        "CALC": "Alcohol consumption (CALC)",
        "MTRANS": "Transportation (MTRANS)",
    }
    return replacements.get(name, name.replace("_", " ").strip())


def compute_bmi(height_value: float, weight_value: float) -> float | None:
    """Compute BMI from height and weight (meters or centimeters)."""
    try:
        h = float(height_value)
        w = float(weight_value)
        if h <= 0 or w <= 0:
            return None
        h_m = (h / 100.0) if h > 3.0 else h
        return w / (h_m ** 2)
    except Exception:
        return None


def top_k(classes, probs, k=5):
    """Return top-k (label, prob) pairs."""
    probs = np.asarray(probs)
    idx = np.argsort(probs)[::-1][:k]
    return [(str(classes[i]), float(probs[i])) for i in idx]


# ----------------------------
# Layout
# ----------------------------
left, right = st.columns([1.2, 1.0])

with st.sidebar:
    st.header("Model")
    st.write(f"Recommended: {schema.get('recommended_model', 'Unknown')}")

    # Use schema classes if present; otherwise fall back safely
    classes_from_schema = schema.get("classes", [])
    n_classes = len(classes_from_schema) if isinstance(classes_from_schema, list) else 0
    if n_classes <= 0:
        n_classes = 7  # safe default for this obesity dataset

    st.write(f"Number of classes: {n_classes}")

    st.divider()

    # Dynamic Top-K: never allow choosing more than available classes
    top_k_n = st.slider(
        "Top-K results",
        min_value=1,
        max_value=n_classes,
        value=min(5, n_classes),
        step=1,
    )

    show_all = st.checkbox("Show all class probabilities", value=False)

    # Optional: if show_all is on, Top-K becomes "all classes" for consistent display
    if show_all:
        top_k_n = n_classes


# ----------------------------
# Inputs
# ----------------------------
with left:
    st.subheader("Input features")

    numeric_features = schema.get("numeric_features", {})
    categorical_features = schema.get("categorical_features", {})

    user_inputs = {}

    with st.form("input_form", clear_on_submit=False):
        # Numeric section
        if numeric_features:
            st.markdown("Numeric")
            cols = st.columns(2)
            i = 0
            for feat, info in numeric_features.items():
                vmin = float(info["min"])
                vmax = float(info["max"])
                vdef = float(info["median"])

                use_int = feat in INT_FEATURE_HINTS

                with cols[i % 2]:
                    if use_int:
                        # Integer widget + integer bounds
                        imin = int(np.floor(vmin))
                        imax = int(np.ceil(vmax))
                        idef = int(round(vdef))
                        idef = max(imin, min(imax, idef))

                        user_inputs[feat] = st.number_input(
                            label=pretty_label(feat),
                            min_value=imin,
                            max_value=imax,
                            value=idef,
                            step=1,
                        )
                    else:
                        # Float widget with a reasonable step
                        span = vmax - vmin
                        step = 0.1 if span <= 5 else (0.5 if span <= 50 else 1.0)

                        user_inputs[feat] = st.number_input(
                            label=pretty_label(feat),
                            min_value=vmin,
                            max_value=vmax,
                            value=vdef,
                            step=step,
                            format="%.3f" if step < 1 else "%.2f",
                        )
                i += 1

        # Categorical section
        if categorical_features:
            st.markdown("Categorical")
            cols = st.columns(2)
            j = 0
            for feat, info in categorical_features.items():
                options = info["options"]
                default = info["default"]
                default_index = options.index(default) if default in options else 0

                with cols[j % 2]:
                    user_inputs[feat] = st.selectbox(
                        label=pretty_label(feat),
                        options=options,
                        index=default_index,
                    )
                j += 1

        st.divider()
        submitted = st.form_submit_button("Predict")


# ----------------------------
# Results
# ----------------------------
with right:
    st.subheader("Prediction result")

    # Show BMI computed (helpful demo)
    bmi_val = None
    if "Height" in user_inputs and "Weight" in user_inputs:
        bmi_val = compute_bmi(user_inputs["Height"], user_inputs["Weight"])

    if bmi_val is not None:
        st.metric("BMI (computed)", f"{bmi_val:.2f}")

    if submitted:
        # Cast ints explicitly where requested (Age as int)
        for feat in list(user_inputs.keys()):
            if feat in INT_FEATURE_HINTS:
                user_inputs[feat] = int(user_inputs[feat])

        x_in = pd.DataFrame([user_inputs])

        # Predict probabilities (RandomForest supports predict_proba)
        probs = model.predict_proba(x_in)[0]
        classes = model.classes_

        pred_idx = int(np.argmax(probs))
        pred_label = str(classes[pred_idx])
        pred_prob = float(probs[pred_idx])

        st.success(f"Predicted obesity level: {pred_label}")
        st.caption(f"Top-1 probability: {pred_prob:.3f}")

        # Top-K table
        k = int(min(top_k_n, len(classes)))
        topk = top_k(classes, probs, k=k)
        df_top = pd.DataFrame(topk, columns=["Class", "Probability"])

        st.markdown("Top results")
        st.dataframe(df_top, use_container_width=True, hide_index=True)

        st.markdown("Probability chart (Top-K)")
        st.bar_chart(df_top.set_index("Class"))

        if show_all:
            st.markdown("All class probabilities")
            df_all = (
                pd.DataFrame({"Class": classes, "Probability": probs})
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(df_all, use_container_width=True, hide_index=True)

    else:
        st.info("Fill in the inputs and click Predict.")