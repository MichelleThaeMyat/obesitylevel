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

# Hide "Press Enter to submit form" helper text and remove red focus border
st.markdown("""
<style>
    div[data-testid="InputInstructions"] {
        display: none;
    }
    /* Remove red focus border from all inputs */
    input:focus, input:active {
        border-color: rgba(250, 250, 250, 0.2) !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="input"]:focus,
    div[data-baseweb="input"]:active {
        border-color: rgba(250, 250, 250, 0.2) !important;
        box-shadow: none !important;
        outline: none !important;
    }
    /* Target Streamlit's specific wrapper */
    .stNumberInput > div > div {
        border-color: rgba(250, 250, 250, 0.2) !important;
    }
    .stNumberInput > div > div:focus-within {
        border-color: rgba(250, 250, 250, 0.2) !important;
        box-shadow: none !important;
    }
    /* Override any red/primary color borders */
    *:focus {
        outline: none !important;
    }
    [data-baseweb="base-input"] {
        border-color: rgba(250, 250, 250, 0.2) !important;
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

# Feature units mapping
FEATURE_UNITS = {
    "Age": "years",
    "Height": "cm",
    "Weight": "lb",
    "FCVC": "scale 1-3",
    "NCP": "meals/day",
    "CH2O": "liters/day",
    "FAF": "hours/week",
    "TUE": "hours/day",
}

# Feature descriptions for help text
FEATURE_HELP = {
    "Age": "Your age in years",
    "Height": "Your height in centimeters (e.g., 175 cm)",
    "Weight": "Your weight in pounds (lb)",
    "FCVC": "1 = Low, 2 = Medium, 3 = High vegetable consumption",
    "NCP": "Number of main meals you eat per day (1-4)",
    "CH2O": "1 = Low, 2 = Moderate, 3 = High water intake",
    "FAF": "0 = No exercise, 1 = Low, 2 = Moderate, 3 = High activity",
    "TUE": "0 = Low, 1 = Moderate, 2 = High screen time",
    "Gender": "Your biological gender",
    "family_history_with_overweight": "Does your family have a history of overweight?",
    "FAVC": "Do you frequently eat high-calorie foods (fast food, junk food)?",
    "CAEC": "How often do you snack between meals?",
    "SMOKE": "Do you smoke?",
    "SCC": "Do you monitor/track your calorie intake?",
    "CALC": "How often do you consume alcohol?",
    "MTRANS": "Your main mode of transportation",
}


def pretty_label(name: str) -> str:
    """Convert raw feature names into readable labels with units."""
    base_labels = {
        "family_history_with_overweight": "Family history with overweight",
        "FAVC": "High-calorie food consumption",
        "FCVC": "Vegetable consumption frequency",
        "NCP": "Number of main meals",
        "CAEC": "Eating between meals",
        "SMOKE": "Smoker",
        "CH2O": "Daily water intake",
        "SCC": "Calorie monitoring",
        "FAF": "Physical activity frequency",
        "TUE": "Time using technology",
        "CALC": "Alcohol consumption",
        "MTRANS": "Transportation",
    }
    label = base_labels.get(name, name.replace("_", " ").strip())
    
    # Add unit if available
    if name in FEATURE_UNITS:
        label = f"{label} ({FEATURE_UNITS[name]})"
    
    return label


def compute_bmi(height_cm: float, weight_lb: float) -> float | None:
    """Compute BMI from height in cm and weight in lb."""
    try:
        h_cm = float(height_cm)
        w_lb = float(weight_lb)
        if h_cm <= 0 or w_lb <= 0:
            return None
        # Convert to meters and kg for BMI formula
        h_m = h_cm / 100.0
        w_kg = w_lb * 0.453592
        return w_kg / (h_m ** 2)
    except Exception:
        return None


def lb_to_kg(lb: float) -> float:
    """Convert pounds to kilograms."""
    return lb * 0.453592


def cm_to_m(cm: float) -> float:
    """Convert centimeters to meters."""
    return cm / 100.0


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

    # Collect Height and Weight first (outside form for real-time BMI calculation)
    st.markdown("**Body Measurements**")
    col_hw = st.columns(2)
    
    # Get Height info from schema (convert from meters to cm for display)
    height_info = numeric_features.get("Height", {"min": 1.0, "max": 2.5, "median": 1.7})
    weight_info = numeric_features.get("Weight", {"min": 30, "max": 200, "median": 70})
    
    # Convert schema values (meters) to cm for UI
    height_min_cm = float(height_info["min"]) * 100 if float(height_info["min"]) < 10 else float(height_info["min"])
    height_max_cm = float(height_info["max"]) * 100 if float(height_info["max"]) < 10 else float(height_info["max"])
    height_default_cm = float(height_info["median"]) * 100 if float(height_info["median"]) < 10 else float(height_info["median"])
    
    # Convert schema values (kg) to lb for UI
    weight_min_lb = float(weight_info["min"]) * 2.20462
    weight_max_lb = float(weight_info["max"]) * 2.20462
    weight_default_lb = float(weight_info["median"]) * 2.20462
    
    with col_hw[0]:
        height_cm = st.number_input(
            label="Height (cm)",
            min_value=height_min_cm,
            max_value=height_max_cm,
            value=height_default_cm,
            step=1.0,
            format="%.0f",
            help="Your height in centimeters (e.g., 175 cm)",
            key="height_input",
        )
        # Store in meters for model
        user_inputs["Height"] = cm_to_m(height_cm)
    
    with col_hw[1]:
        weight_lb = st.number_input(
            label="Weight (lb)",
            min_value=weight_min_lb,
            max_value=weight_max_lb,
            value=weight_default_lb,
            step=1.0,
            format="%.0f",
            help="Your weight in pounds (lb)",
            key="weight_input",
        )
        # Store in kg for model
        user_inputs["Weight"] = lb_to_kg(weight_lb)
    
    # Auto-calculate BMI (using cm and lb inputs)
    bmi_calculated = compute_bmi(height_cm, weight_lb)
    if bmi_calculated is not None:
        user_inputs["BMI"] = bmi_calculated
        
        # Determine BMI category
        if bmi_calculated < 18.5:
            bmi_category = "Underweight"
            bmi_color = "blue"
        elif bmi_calculated < 25:
            bmi_category = "Normal"
            bmi_color = "green"
        elif bmi_calculated < 30:
            bmi_category = "Overweight"
            bmi_color = "orange"
        else:
            bmi_category = "Obese"
            bmi_color = "red"
        
        weight_kg = lb_to_kg(weight_lb)
        height_m = cm_to_m(height_cm)
        st.markdown(f"**BMI (kg/m²):** `{bmi_calculated:.2f}` — :{bmi_color}[{bmi_category}]")
        st.caption(f"Auto-calculated: {weight_kg:.1f} kg / {height_m:.2f}² m = {bmi_calculated:.2f}")

    with st.form("input_form", clear_on_submit=False):
        # Numeric section (excluding Height, Weight, BMI which are handled above)
        other_numeric = {k: v for k, v in numeric_features.items() if k not in ["Height", "Weight", "BMI"]}
        
        if other_numeric:
            st.markdown("**Other Numeric Features**")
            cols = st.columns(2)
            i = 0
            for feat, info in other_numeric.items():
                vmin = float(info["min"])
                vmax = float(info["max"])
                vdef = float(info["median"])

                use_int = feat in INT_FEATURE_HINTS

                with cols[i % 2]:
                    help_text = FEATURE_HELP.get(feat, None)
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
                            help=help_text,
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
                            help=help_text,
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
                    help_text = FEATURE_HELP.get(feat, None)
                    user_inputs[feat] = st.selectbox(
                        label=pretty_label(feat),
                        options=options,
                        index=default_index,
                        help=help_text,
                    )
                j += 1

        st.divider()
        submitted = st.form_submit_button("Predict")


# ----------------------------
# Results
# ----------------------------
with right:
    st.subheader("Prediction result")

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