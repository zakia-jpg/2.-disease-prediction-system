import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Disease Prediction System", page_icon="🩺")

# ----------------------------
# File Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

USERS_FILE = os.path.join(BASE_DIR, "users.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "history.csv")

# Create files if not exist
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["name", "password", "age", "email"]).to_csv(USERS_FILE, index=False)

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["name", "disease", "symptoms"]).to_csv(HISTORY_FILE, index=False)

# ----------------------------
# Load ML Files
# ----------------------------
with open(os.path.join(BASE_DIR, "disease_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

df = pd.read_csv(os.path.join(BASE_DIR, "disease_prediction_dataset_10000_rows_23_columns.csv"))

# ----------------------------
# Session
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ----------------------------
# MAIN TITLE
# ----------------------------
st.title("🩺 DISEASE PREDICTION SYSTEM")

# ----------------------------
# AUTH PAGES
# ----------------------------
if not st.session_state.logged_in:

    choice = st.radio("Select Option", ["Login", "Sign Up"])

    users_df = pd.read_csv(USERS_FILE)

    # -------- SIGN UP --------
    if choice == "Sign Up":
        st.subheader("Create Account")

        name = st.text_input("Username")
        password = st.text_input("Password", type="password")
        age = st.number_input("Age", 1, 120)
        email = st.text_input("Email")

        if st.button("Sign Up"):
            if name in users_df["name"].values:
                st.error("User already exists!")
            else:
                new_user = pd.DataFrame([[name, password, age, email]],
                                        columns=["name", "password", "age", "email"])
                new_user.to_csv(USERS_FILE, mode="a", header=False, index=False)
                st.success("Account created! Please login.")

    # -------- LOGIN --------
    elif choice == "Login":
        st.subheader("Login")

        name = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = users_df[(users_df["name"] == name) & (users_df["password"] == password)]

            if not user.empty:
                st.session_state.logged_in = True
                st.session_state.user_name = name
                st.success("Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid credentials ❌")

# ----------------------------
# MAIN APP
# ----------------------------
else:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "History", "Logout"])

    # -------- PREDICTION --------
    if page == "Prediction":
        st.subheader(f"Welcome {st.session_state.user_name} 👋")

        selected_symptoms = st.multiselect("Select Symptoms", feature_columns)

        if st.button("Predict"):
            if selected_symptoms:

                input_data = np.zeros(len(feature_columns))
                for symptom in selected_symptoms:
                    input_data[feature_columns.index(symptom)] = 1

                prediction = model.predict([input_data])
                disease = le.inverse_transform(prediction)[0]

                st.success(f"Predicted Disease: {disease}")

                # Save history
                history = pd.DataFrame([[
                    st.session_state.user_name,
                    disease,
                    ", ".join(selected_symptoms)
                ]], columns=["name", "disease", "symptoms"])

                history.to_csv(HISTORY_FILE, mode="a", header=False, index=False)

                # Show precautions
                precautions = df[df["Disease"] == disease][[
                    "Precaution_1",
                    "Precaution_2",
                    "Precaution_3",
                    "Precaution_4"
                ]].iloc[0]

                st.subheader("Precautions")
                for p in precautions:
                    st.write("•", p)

            else:
                st.warning("Select symptoms")

    # -------- HISTORY --------
    elif page == "History":
        st.subheader("Your Prediction History")

        history_df = pd.read_csv(HISTORY_FILE)
        user_history = history_df[history_df["name"] == st.session_state.user_name]

        if not user_history.empty:
            st.dataframe(user_history)
        else:
            st.info("No history found")

    # -------- LOGOUT --------
    elif page == "Logout":
        st.session_state.logged_in = False
        st.success("Logged out successfully")
        st.rerun()