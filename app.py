import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@st.cache_resource
def train_model():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "titanic.csv"))
    X = df[["Pclass", "Sex", "Age", "Fare"]]
    y = df["Survived"]
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(), ["Pclass", "Sex"]),
        ("num", "passthrough", ["Age", "Fare"])
    ])
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    return model

model = train_model()

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 30)
fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=32.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[pclass, sex, age, fare]],
                            columns=["Pclass", "Sex", "Age", "Fare"])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ This passenger would likely **SURVIVE**.")
    else:
        st.error("❌ This passenger would likely **NOT survive**.")
