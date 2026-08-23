import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a tourism package based on their profile and trip preferences.
Enter the customer details below to get a prediction.
""")

Type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
age             = st.number_input("Age", 18, 100, 30)
city_tier       = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 0.0, 100.0, 15.0, 0.1)
occupation      = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender          = st.selectbox("Gender", ["Female", "Male"])
num_person      = st.number_input("Number of Persons Visiting", 1, 10, 2)
num_followups   = st.number_input("Number of Follow-ups", 1, 10, 4)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
pref_prop_star  = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status  = st.selectbox("Marital Status", ["Single", "Divorced", "Married"])
num_trips       = st.number_input("Number of Trips Annually", 1, 20, 3)
passport        = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
satisfaction    = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car         = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
num_children    = st.number_input("Number of Children Visiting (under 5)", 0, 5, 0)
designation     = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
monthly_income  = st.number_input("Monthly Income", 0.0, 150000.0, 20000.0, 500.0)

input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "NumberOfPersonVisiting": num_person,
    "PreferredPropertyStar": pref_prop_star,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": satisfaction,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_of_pitch,
    "TypeofContact": Type_of_contact,
    "Occupation": occupation,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "Designation": designation,
    "ProductPitched": product_pitched,
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Package Purchased" if prediction == 1 else "Package Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
