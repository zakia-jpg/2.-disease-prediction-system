import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd


# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="centered")


# ----------------------------
# Load Files Safely
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "disease_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

csv_path = os.path.join(BASE_DIR, "disease_prediction_dataset_10000_rows_23_columns.csv")
df = pd.read_csv(csv_path)


# ----------------------------
# Session State
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("🧭 Navigation")

if st.session_state.logged_in:
    page = st.sidebar.radio("Go to", ["Prediction", "Logout"])
else:
    page = "Login"


# ----------------------------
# LOGIN PAGE
# ----------------------------
if page == "Login":

    st.title("🩺 Disease Prediction System")
    st.subheader("User Login")

    name = st.text_input("Enter Your Name")
    email = st.text_input("Enter Your Email")
    age = st.number_input("Enter Your Age", min_value=1, max_value=120)

    if st.button("Login"):
        if name and email and age:
            st.session_state.logged_in = True
            st.session_state.user_name = name
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Please fill all details")


# ----------------------------
# PREDICTION PAGE
# ----------------------------
elif page == "Prediction":

    st.title("🧬 Disease Prediction")
    st.write(f"Welcome, {st.session_state.user_name} 👋")

    st.markdown("### Select Your Symptoms")

    selected_symptoms = st.multiselect(
        "Choose Symptoms",
        feature_columns
    )

    if st.button("Predict Disease"):

        if selected_symptoms:

            input_data = np.zeros(len(feature_columns))

            for symptom in selected_symptoms:
                index = feature_columns.index(symptom)
                input_data[index] = 1

            input_data = input_data.reshape(1, -1)

            prediction = model.predict(input_data)
            disease = le.inverse_transform(prediction)[0]

            st.success(f"🩺 Predicted Disease: {disease}")

            precautions = df[df["Disease"] == disease][[
                "Precaution_1",
                "Precaution_2",
                "Precaution_3",
                "Precaution_4"
            ]].iloc[0]

            st.markdown("### 🛡️ Recommended Precautions")

            for p in precautions:
                st.write("•", p)

            st.info("⚠️ This is a prediction system. Please consult a doctor for medical advice.")

        else:
            st.warning("Please select at least one symptom.")


# ----------------------------
# LOGOUT
# ----------------------------
elif page == "Logout":

    st.session_state.logged_in = False
    st.success("Logged out successfully.")
    st.rerun()