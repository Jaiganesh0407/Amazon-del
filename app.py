import streamlit as st
import joblib
import pandas as pd
import numpy as np
from geopy.distance import great_circle # Used for distance calculation
import json

# --- 1. Load Artifacts ---

@st.cache_resource
def load_model_artifacts():
    """Loads the trained model and feature list."""
    try:
        # Load the trained model
        model = joblib.load('best_delivery_model.pkl')
        # Load the feature names list
        with open('model_features.json', 'r') as f:
            feature_names = json.load(f)
        return model, feature_names
    except FileNotFoundError:
        st.error("Error: Model files not found. Ensure 'best_delivery_model.pkl' and 'model_features.json' are in the directory.")
        return None, None

model, feature_names = load_model_artifacts()

if model is None:
    st.stop()
    
# --- 2. Feature Engineering Helper ---

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance using geopy.great_circle (km)."""
    try:
        start = (lat1, lon1)
        end = (lat2, lon2)
        return great_circle(start, end).km
    except:
        return 0.0

# --- 3. Streamlit Interface ---

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="wide")
st.title("📦 Amazon Delivery Time Predictor")
st.markdown("---")
st.markdown("### Input Delivery Details")

# Define input fields based on the model's training data
col1, col2 = st.columns(2)

# Column 1 Inputs (Agent, Time)
with col1:
    st.subheader("Agent & Time Details")
    agent_age = st.slider("Agent Age (Years)", 20, 40, 30)
    agent_rating = st.slider("Agent Rating (1.0 to 5.0)", 2.5, 5.0, 4.5, 0.1)
    time_to_pickup_min = st.number_input("Time from Order to Pickup (min)", min_value=1, max_value=30, value=10)
    order_hour = st.slider("Order Hour (24h format)", 0, 23, 15)
    
    # Mapping for Day_of_Week feature
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    day_of_week_str = st.selectbox("Day of Week", list(day_mapping.keys()), index=4)
    day_of_week = day_mapping[day_of_week_str]

# Column 2 Inputs (Conditions, Location)
with col2:
    st.subheader("Condition & Location")
    
    weather = st.selectbox("Weather Condition", ['Sunny', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog', 'Windy'])
    traffic = st.selectbox("Traffic Condition", ['Low', 'Medium', 'High', 'Jam'])
    vehicle = st.selectbox("Vehicle Type", ['motorcycle', 'scooter', 'van'])
    area = st.selectbox("Area Type", ['Urban', 'Metropolitian'])
    
    st.markdown("##### GPS Coordinates (Store to Drop)")
    store_lat = st.number_input("Store Latitude", value=12.914264, format="%.6f")
    store_lon = st.number_input("Store Longitude", value=77.678400, format="%.6f")
    drop_lat = st.number_input("Drop Latitude", value=12.924264, format="%.6f")
    drop_lon = st.number_input("Drop Longitude", value=77.688400, format="%.6f")
    
    category = st.selectbox("Category", ['Clothing', 'Electronics', 'Sports', 'Cosmetics', 'Toys', 'Snacks', 'Jewelry', 'Grocery', 'Outdoors', 'Books', 'Home Supplies', 'Skincare', 'Apparel', 'Shoes'])

# --- 4. Prediction Logic ---

if st.button("PREDICT DELIVERY TIME", type="primary"):
    
    # 1. Calculate derived features
    distance_km = calculate_distance(store_lat, store_lon, drop_lat, drop_lon)
    
    # 2. Prepare base input data
    input_data = {
        'Agent_Age': agent_age,
        'Agent_Rating': agent_rating,
        'Time_to_Pickup_min': time_to_pickup_min,
        'Distance_km': distance_km,
        'Order_Hour': order_hour,
        'Day_of_Week': day_of_week,
    }
    
    # 3. Apply One-Hot Encoding to categorical inputs
    # Use the stripped and cleaned column names as used in training
    input_data[f'Weather_{weather.strip()}_nan'] = 0 # Placeholder for drop_first=True column name (e.g. Weather_Sunny_nan)
    if f'Weather_{weather.strip()}' in feature_names:
        input_data[f'Weather_{weather.strip()}'] = 1
    if f'Traffic_{traffic.strip()}' in feature_names:
        input_data[f'Traffic_{traffic.strip()}'] = 1
    if f'Vehicle_{vehicle.strip()}' in feature_names:
        input_data[f'Vehicle_{vehicle.strip()}'] = 1
    if f'Area_{area.strip()}' in feature_names:
        input_data[f'Area_{area.strip()}'] = 1
    if f'Category_{category.strip()}' in feature_names:
        input_data[f'Category_{category.strip()}'] = 1
    
    # 4. Create DataFrame and align columns with the model's expected features
    input_df = pd.DataFrame([input_data]).reindex(columns=feature_names, fill_value=0)
    
    # 5. Make prediction
    try:
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        st.subheader("✅ Prediction Successful")
        
        # Display the prediction range 
        st.success(f"**Estimated Delivery Time: {int(prediction):} to {int(prediction + 20):} minutes**")
        st.info(f"Calculated distance: **{distance_km:.2f} km**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check the input values and ensure all required files are correctly loaded.")
