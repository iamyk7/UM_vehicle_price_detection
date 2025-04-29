import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('vehicle_price_predictor.joblib')

model = load_model()

# App title and description
st.title('🚗 Vehicle Price Predictor')
st.markdown("""
Enter your vehicle details below to get an estimated price prediction.
""")

# Input form
with st.form("vehicle_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        make = st.text_input("Make (e.g., Toyota, Ford)", "")
        model_name = st.text_input("Model (e.g., Camry, F-150)", "")
        year = st.number_input("Year", min_value=1900, max_value=2023, value=2015)
        mileage = st.number_input("Mileage", min_value=0, value=50000)
    
    with col2:
        cylinders = st.number_input("Cylinders", min_value=0, max_value=12, value=4)
        doors = st.number_input("Doors", min_value=2, max_value=6, value=4)
        fuel_type = st.selectbox("Fuel Type", 
                               ["Gasoline", "Diesel", "Electric", "Hybrid"])
        transmission = st.selectbox("Transmission", 
                                  ["Automatic", "Manual", "CVT"])
        body_type = st.selectbox("Body Type", 
                               ["Sedan", "SUV", "Truck", "Coupe", "Hatchback"])
        drivetrain = st.selectbox("Drivetrain", 
                                ["FWD", "RWD", "AWD", "4WD"])
    
    submitted = st.form_submit_button("Predict Price")

# When form is submitted
if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'year': year,
        'mileage': mileage,
        'cylinders': cylinders,
        'doors': doors,
        'make': make,
        'model': model_name,
        'fuel': fuel_type,
        'transmission': transmission,
        'body': body_type,
        'drivetrain': drivetrain
    }])
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.success(f"### Predicted Price: ${prediction:,.2f}")
        
        # Show input details
        with st.expander("See details"):
            st.write("**Vehicle Details:**")
            st.json({
                "Make": make,
                "Model": model_name,
                "Year": year,
                "Mileage": f"{mileage:,} miles",
                "Cylinders": cylinders,
                "Doors": doors,
                "Fuel Type": fuel_type,
                "Transmission": transmission,
                "Body Type": body_type,
                "Drivetrain": drivetrain
            })
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check your inputs and try again.")

# Add some info about the model
st.sidebar.markdown("""
### About This Predictor
This tool uses a machine learning model trained on historical vehicle data to estimate prices.

**How to use:**
1. Fill in all vehicle details
2. Click "Predict Price"
3. View your estimated price

**Note:** Predictions are estimates only and actual market prices may vary.
""")