import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Enoch1359/Tourism_model", filename="best_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title(" Tourism Package Prediction")
st.write("""
This application predicts the likelihood of customer buying tourism package.
""")


age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
typeof_contact = st.selectbox("TypeofContact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("CityTier", [1, 2, 3])
occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
gender = st.selectbox("Gender", ['Female', 'Male'])
num_person_visiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=5, value=2, step=1)
preferred_star = st.selectbox("PreferredPropertyStar", [ 3.0, 4.0 ,5.0])
marital_status = st.selectbox("MaritalStatus", ['Single', 'Divorced', 'Married'])
num_trips = st.number_input("NumberOfTrips", min_value=0, max_value=50, value=2, step=1)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("OwnCar", [0, 1])
num_children = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0, step=1)
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
monthly_income = st.number_input("MonthlyIncome", min_value=1000, max_value=1000000, value=20000, step=1000)

# Customer Interaction Data
pitch_score = st.slider("PitchSatisfactionScore", min_value=1, max_value=5, value=3)
product_pitched = st.selectbox("ProductPitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
num_followups = st.number_input("NumberOfFollowups", min_value=0, max_value=20, value=2, step=1)
duration_pitch = st.number_input("DurationOfPitch", min_value=0, max_value=300, value=15, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_person_visiting,
    'PreferredPropertyStar': preferred_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': num_followups,
    'DurationOfPitch': duration_pitch
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Machine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
