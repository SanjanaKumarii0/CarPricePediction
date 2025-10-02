import streamlit as st
import pickle
import numpy as np
# Custom Background & Styling (Teal Green Theme)
# ---------------------------
page_bg = """
<style>
.stApp {
    background-color: #0D9488; /* Teal Green background */
    color: white;
}
.stButton button {
    background-color: #14B8A6; /* Lighter Teal button */
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.5rem 1.5rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------
# Load Trained Model
# ---------------------------
model = pickle.load(open("model.pkl", "rb"))  # apna model file ka naam

st.title("🚗 Car Price Prediction App")
st.write("Is app se aap apni gaadi ki *selling price* predict kar sakte ho.")

# ---------------------------
# User Inputs
# ---------------------------
st.header("Enter Car Details:")

year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, max_value=50.0, value=5.0)
kms_driven = st.number_input("Kms Driven", min_value=0, max_value=1000000, value=50000)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.number_input("Number of Previous Owners", min_value=0, max_value=5, value=0)

# Encoding categorical values (must match training LabelEncoder)
fuel_map = {"CNG": 0, "Diesel": 1, "Petrol": 2}
seller_map = {"Dealer": 0, "Individual": 1}
transmission_map = {"Automatic": 0, "Manual": 1}

fuel_val = fuel_map[fuel_type]
seller_val = seller_map[seller_type]
trans_val = transmission_map[transmission]

# Correct feature order as training
features = np.array([[year, present_price, kms_driven, owner, fuel_val, seller_val, trans_val]])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Selling Price"):
    prediction = model.predict(features)
    st.subheader("💰 Predicted Selling Price:")
    st.success(f"₹ {round(prediction[0],2)} lakhs")
