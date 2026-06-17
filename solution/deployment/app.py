import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="asifaddicted/superkart-sales-forecast-model", filename="best_sales_forecast_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("SuperKart Sales Forecast")
st.write("""
This application predicts the expected **Sales Forecast** for a SuperKart product
based on characteristics such as product weight, type, MRP, store size, and location.
Please enter the details below to get a prediction.
""")

# User input
Product_Weight = st.number_input("Product Weight", min_value=0.004, max_value=0.298, value=0.004, step=0.001)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.001, max_value=10.99, value=0.001, step=0.001)

Product_Type = st.selectbox("Product Type", ["Household", "Starchy Foods", "Dairy", "Snack Foods", "Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Meat", "Fruits and Vegetables", "Soft Drinks", "Others", "Health and Hygiene", "Breads", "Seafood", "Breakfast"])
Product_MRP = st.number_input("Product MRP", min_value=0.1, max_value=9999.0, value=1.0, step=1.0)
Store_Establishment_Year = st.number_input("Store Establishment Year", min_value=1900, max_value=2026, value=1900, step=1)
Store_Size = st.selectbox("Store Size", ["High", "Medium", "Small"])
Store_Location_City_Type = st.selectbox("Store City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_Type': Product_Type,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

# Predict button
if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Sales: **${prediction:,.2f} USD**")
