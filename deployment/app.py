import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="verma89/tourismprjtt", filename="best_tourism_package_taker_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Sales Predictor")
st.write("""
### Wellness Tourism Package Predictor
This application helps the marketing team at **"Visit with Us"** identify potential customers for the new Wellness Tourism Package.

By analyzing customer demographics and past interactions—such as **Age**, **Monthly Income**, and **Pitch Satisfaction**—this model predicts whether a customer is likely to purchase the package (**1**) or not (**0**).
""")


# User Input Section
st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    duration_pitch = st.number_input("Duration of Pitch (min)", min_value=1, max_value=120, value=15)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=25000)
    num_trips = st.number_input("Number of Trips per Year", min_value=1, max_value=20, value=3)

with col2:
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    passport = st.selectbox("Has Passport?", [0, 1], help="0: No, 1: Yes")
    own_car = st.selectbox("Owns a Car?", [0, 1], help="0: No, 1: Yes")
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# Categorical Dropdowns (Must match your LabelEncoder classes)
st.subheader("Demographics & Interaction")
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

# Additional numeric features
num_person = st.number_input("Number of Persons Visiting", 1, 5, 2)
num_followups = st.number_input("Number of Follow-ups", 1, 6, 3)
num_children = st.number_input("Number of Children Visiting", 0, 3, 0)
preferred_star = st.slider("Preferred Property Star Rating", 3, 5, 3)


# Assemble input into DataFrame
# CRITICAL: The keys here must match your training features exactly
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_person,
    'NumberOfFollowups': num_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Predict button
if st.button("Predict Purchase Likelihood"):
    # The model pipeline handles scaling/encoding automatically
    prediction = model.predict(input_data)[0]
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("Target this customer! They are **LIKELY** to purchase the Wellness Package.")
    else:
        st.warning("This customer is **UNLIKELY** to purchase the package at this time.")
