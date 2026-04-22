import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Titanic Survival Predictor', page_icon='🚢', layout='centered')

@st.cache_resource
def load_model():
    return joblib.load('titanic_model.pkl')

model = load_model()

st.title('🚢 Titanic Survival Predictor')
st.markdown('Enter passenger details to predict survival.')
st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        'Passenger Class',
        options=[1, 2, 3],
        format_func=lambda x: {1:'1st Class', 2:'2nd Class', 3:'3rd Class'}[x]
    )
    age = st.slider('Age', min_value=1, max_value=80, value=30)

with col2:
    sex = st.selectbox('Sex', options=['male', 'female'])
    fare = st.number_input('Fare Paid', min_value=0.0, max_value=600.0, value=32.0, step=1.0)

st.divider()

if st.button('Predict Survival', use_container_width=True):
    input_df = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [float(age)],
        'Fare': [float(fare)]
    })

    try:
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        st.subheader('Prediction Result')

        if prediction == 1:
            st.success('This passenger would likely SURVIVE!')
        else:
            st.error('This passenger would likely NOT survive.')

        st.metric('Survival Probability', f'{probability:.1%}')

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.divider()
st.caption('INFO 648 - CRISP-DM Mini Project')
