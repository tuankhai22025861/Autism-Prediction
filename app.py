import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders.pkl")

# Giao diện
st.title("🧠 Autism Prediction Web App")

with st.form("prediction_form"):
    st.subheader("Please fill in the following information:")

    scores = {f"A{i}_Score": st.selectbox(f"A{i}_Score", [0, 1], key=f"a{i}") for i in range(1, 10+1)}
    age = st.number_input("Age", min_value=1, max_value=120, step=1)

    gender = st.selectbox("Gender", ['f', 'm'])
    jaundice = st.selectbox("Jaundice (yes/no)", ['no', 'yes'])
    austim = st.selectbox("Family member with autism? (yes/no)", ['no', 'yes'])
    relation = st.selectbox("Relation to individual", ['Self', 'Others'])
    result = st.number_input("Screening Test Score (result)", min_value=0, max_value=20, step=1)

    submit = st.form_submit_button("Predict")

if submit:
    input_dict = {
        **scores,
        "age": age,
        "gender": gender,
        "jaundice": jaundice,
        "austim": austim,
        "relation": relation,
        "result": result
    }

    df = pd.DataFrame([input_dict])

    # Áp dụng encoding
    for col in encoders:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df.values)[0]
    result_text = "⚠️ Likely ASD" if prediction == 1 else "✅ Unlikely ASD"
    st.success(f"Prediction Result: {result_text}")
