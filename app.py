# streamlit_app_obesity_ui_preferred_v2.py
# Clean Streamlit UI (no emojis) for Obesity Level Prediction
# - User inputs: Height (cm), Weight (lb)
# - Internal: converts to meters/kg + computes BMI (kg/m^2)
# - IMPORTANT UI change:
#     * BMI is shown as a neutral metric ONLY (no "Overweight/Normal" label)
#     * Adds a note to avoid confusion with the ML prediction result
# - Removes "Press Enter to apply" helper text and red focus border
# - Top-K + (optional) show all probabilities
# - Matplotlib bar chart to avoid truncated class labels

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Obesity Level Prediction",
    page_icon=None,
    layout="wide",
)

# ----------------------------
# CSS: hide helper text + remove focus red border
# ----------------------------
st.markdown(
    """
<style>
/* Hide Streamlit helper text like "Press Enter to apply" */
div[data-testid="InputInstructions"] { display: none !important; }

/* Remove focus outlines / red borders */
input:focus, input:active, input:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="input"]:focus,
div[data-baseweb="input"]:active {
    border-color: rgba(255,255,255,0.18) !important;
    box-shadow: none !important;
    outline: none !important;
}
*:focus { outline: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

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
INT_FEATURE_HINTS = {"Age"}  # features forced to int


def lb_to_kg(lb: float) -> float:
    """Convert pounds to kilograms."""
    return float(lb) * 0.45359237


def cm_to_m(cm: float) -> float:
    """Convert centimeters to meters."""
    return float(cm) / 100.0


def compute_bmi_cm_lb(height_cm: float, weight_lb: float) -> float | None:
    """Compute BMI in kg/m^2 from height(cm) and weight(lb)."""
    try:
        h_m = cm_to_m(height_cm)
        w_kg = lb_to_kg(weight_lb)
        if h_m <= 0 or w_kg <= 0:
            return None
        return w_kg / (h_m ** 2)
    except Exception:
        return None


def top_k_pairs(classes, probs, k: int):
    """Return top-k pairs sorted by probability."""
    probs = np.asarray(probs, dtype=float)
    idx = np.argsort(probs)[::-1][:k]
    return [(str(classes[i]), float(probs[i])) for i in idx]


def model_expects(feature_name: str) -> bool:
    """
    Check if the model pipeline expects a given feature.
    If the model exposes feature_names_in_, we can detect features reliably.
    """
    try:
        feats = getattr(model, "feature_names_in_", None)
        if feats is None:
            return False
        return feature_name in list(feats)
    except Exception:
        return False


# Your refined notebook trained with BMI, so this should be True in most cases.
EXPECTS_BMI = model_expects("BMI")


def pretty_label(name: str) -> str:
    """Readable labels for UI."""
    mapping = {
        "family_history_with_overweight": "Family history with overweight",
        "FAVC": "High-calorie food consumption",
        "FCVC": "Vegetable consumption frequency (scale 1-3)",
        "NCP": "Number of main meals (meals/day)",
        "CAEC": "Eating between meals",
        "SMOKE": "Smoker",
        "CH2O": "Daily water intake (liters/day)",
        "SCC": "Calorie monitoring",
        "FAF": "Physical activity frequency (hours/week)",
        "TUE": "Time using technology (hours/day)",
        "CALC": "Alcohol consumption",
        "MTRANS": "Transportation",
        "Age": "Age (years)",
        "Height": "Height (cm)",
        "Weight": "Weight (lb)",
    }
    return mapping.get(name, name.replace("_", " ").strip())


def prettify_class_label(label: str) -> str:
    """Make class labels more readable for charts/tables."""
    # Example: Overweight_Level_II -> Overweight Level II
    return label.replace("_", " ").replace("  ", " ").strip()


# ----------------------------
# Sidebar controls
# ----------------------------
left, right = st.columns([1.25, 1.0])

with st.sidebar:
    st.header("Model")
    st.write(f"Recommended: {schema.get('recommended_model', 'Unknown')}")

    n_classes = len(getattr(model, "classes_", schema.get("classes", [])))
    st.write(f"Number of classes: {n_classes}")

    st.divider()
    top_k_n = st.slider(
        "Top-K results",
        min_value=1,
        max_value=max(1, n_classes),
        value=min(5, n_classes) if n_classes else 5,
        step=1,
    )
    show_all = st.checkbox("Show all class probabilities", value=False)
    if show_all and n_classes:
        top_k_n = n_classes


# ----------------------------
# Inputs
# ----------------------------
numeric_features = schema.get("numeric_features", {})
categorical_features = schema.get("categorical_features", {})

user_inputs = {}

with left:
    st.subheader("Input features")
    st.markdown("**Body Measurements**")

    col_hw = st.columns(2)

    # schema stores Height in meters and Weight in kg → convert to cm/lb for UI
    height_info = numeric_features.get("Height", {"min": 1.0, "max": 2.5, "median": 1.70})
    weight_info = numeric_features.get("Weight", {"min": 30.0, "max": 200.0, "median": 70.0})

    height_min_cm = float(height_info["min"]) * 100.0
    height_max_cm = float(height_info["max"]) * 100.0
    height_def_cm = float(height_info["median"]) * 100.0

    weight_min_lb = float(weight_info["min"]) * 2.20462262
    weight_max_lb = float(weight_info["max"]) * 2.20462262
    weight_def_lb = float(weight_info["median"]) * 2.20462262

    with col_hw[0]:
        height_cm = st.number_input(
            pretty_label("Height"),
            min_value=float(np.floor(height_min_cm)),
            max_value=float(np.ceil(height_max_cm)),
            value=float(round(height_def_cm)),
            step=1.0,
            format="%.0f",
        )
        # Model expects meters
        user_inputs["Height"] = cm_to_m(height_cm)

    with col_hw[1]:
        weight_lb = st.number_input(
            pretty_label("Weight"),
            min_value=float(np.floor(weight_min_lb)),
            max_value=float(np.ceil(weight_max_lb)),
            value=float(round(weight_def_lb)),
            step=1.0,
            format="%.0f",
        )
        # Model expects kg
        user_inputs["Weight"] = lb_to_kg(weight_lb)

    # BMI display (neutral) + include as model input if expected
    bmi_val = compute_bmi_cm_lb(height_cm, weight_lb)
    if bmi_val is not None:
        st.markdown(f"**BMI (kg/m²):** `{bmi_val:.2f}`")
        st.caption("BMI is one indicator. The prediction result is based on all inputs.")
        st.caption(f"Auto-calculated from: {weight_lb:.0f} lb, {height_cm:.0f} cm")

        if EXPECTS_BMI:
            user_inputs["BMI"] = float(bmi_val)

    st.divider()
    st.markdown("**Other Numeric Features**")

    # Other numeric features (exclude Height/Weight/BMI)
    other_numeric = {k: v for k, v in numeric_features.items() if k not in {"Height", "Weight", "BMI"}}

    cols = st.columns(2)
    i = 0
    for feat, info in other_numeric.items():
        vmin = float(info["min"])
        vmax = float(info["max"])
        vdef = float(info["median"])

        with cols[i % 2]:
            if feat in INT_FEATURE_HINTS:
                imin = int(np.floor(vmin))
                imax = int(np.ceil(vmax))
                idef = int(round(vdef))
                idef = max(imin, min(imax, idef))

                user_inputs[feat] = st.number_input(
                    pretty_label(feat),
                    min_value=imin,
                    max_value=imax,
                    value=idef,
                    step=1,
                )
            else:
                span = vmax - vmin
                step = 0.1 if span <= 5 else (0.5 if span <= 50 else 1.0)

                user_inputs[feat] = st.number_input(
                    pretty_label(feat),
                    min_value=vmin,
                    max_value=vmax,
                    value=vdef,
                    step=step,
                    format="%.3f" if step < 1 else "%.2f",
                )
        i += 1

    st.markdown("**Categorical**")
    cols = st.columns(2)
    j = 0
    for feat, info in categorical_features.items():
        options = info["options"]
        default = info["default"]
        default_index = options.index(default) if default in options else 0

        with cols[j % 2]:
            user_inputs[feat] = st.selectbox(
                pretty_label(feat),
                options=options,
                index=default_index,
            )
        j += 1

    st.divider()
    submitted = st.button("Predict")


# ----------------------------
# Results
# ----------------------------
with right:
    st.subheader("Prediction result")

    if submitted:
        # Ensure ints where required
        for f in list(user_inputs.keys()):
            if f in INT_FEATURE_HINTS:
                user_inputs[f] = int(user_inputs[f])

        x_in = pd.DataFrame([user_inputs])

        if not hasattr(model, "predict_proba"):
            st.error("This model does not support predict_proba().")
            st.stop()

        probs = model.predict_proba(x_in)[0]
        classes = model.classes_

        pred_idx = int(np.argmax(probs))
        pred_label = str(classes[pred_idx])
        pred_prob = float(probs[pred_idx])

        st.success(f"Predicted obesity level: {prettify_class_label(pred_label)}")
        st.caption(f"Top-1 probability: {pred_prob:.3f}")

        # Top-K table
        k = int(min(top_k_n, len(classes)))
        topk = top_k_pairs(classes, probs, k)

        # Prettify labels for display
        df_top = pd.DataFrame(
            [(prettify_class_label(c), p) for c, p in topk],
            columns=["Class", "Probability"],
        )

        st.markdown("**Top results**")
        st.dataframe(df_top, use_container_width=True, hide_index=True)

        # Chart with readable labels
        st.markdown("**Probability chart (Top-K)**")
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.bar(df_top["Class"], df_top["Probability"])
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.set_title("Top-K Class Probabilities")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        if show_all:
            st.markdown("**All class probabilities**")
            df_all = (
                pd.DataFrame({"Class": [prettify_class_label(c) for c in classes], "Probability": probs})
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(df_all, use_container_width=True, hide_index=True)

    else:
        st.info("Fill in the inputs and click Predict.")