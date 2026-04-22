import streamlit as st
import pandas as pd
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚢 Titanic Survival Predictor")
st.markdown(
    "Enter a passenger's details below to predict whether they would have "
    "survived the Titanic disaster. Powered by a **Random Forest** classifier "
    "trained on the Titanic dataset."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class (Pclass)",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x],
        help="1st = Upper, 2nd = Middle, 3rd = Lower"
    )
    age = st.slider("Age", min_value=1, max_value=80, value=30, step=1)

with col2:
    sex = st.selectbox("Sex", options=["male", "female"])
    fare = st.number_input(
        "Fare Paid (£)",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=1.0,
        format="%.2f"
    )

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Survival", use_container_width=True, type="primary"):
    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Sex":    [sex],
        "Age":    [float(age)],
        "Fare":   [float(fare)],
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ **This passenger would likely SURVIVE!**")
    else:
        st.error(f"❌ **This passenger would likely NOT survive.**")

    st.metric(
        label="Survival Probability",
        value=f"{probability:.1%}",
        delta=f"{probability - 0.5:+.1%} vs 50% baseline"
    )

    with st.expander("ℹ️ About this prediction"):
        st.markdown(
            f"""
            | Feature | Value |
            |---|---|
            | Passenger Class | {pclass} |
            | Sex | {sex.capitalize()} |
            | Age | {age} years |
            | Fare | £{fare:.2f} |

            The model is a **Random Forest** with 100 decision trees, achieving:
            - Accuracy: 80.9%
            - Precision: 76.2%
            - Recall: 71.6%
            """
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("INFO 648 · CRISP-DM Mini Project · Model: Random Forest · Dataset: Titanic (cleaned)")
