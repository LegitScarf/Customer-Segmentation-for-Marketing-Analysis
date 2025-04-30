# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🧩",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; text-align: center; margin-bottom: 2rem; }
    .stButton button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .segment-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    .segment-0 { background-color: #d4f1f9; color: #05445E; }
    .segment-1 { background-color: #ffecdb; color: #e67e22; }
    .segment-2 { background-color: #e8f4ea; color: #27ae60; }
    .segment-3 { background-color: #f2e4ff; color: #8e44ad; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return kmeans, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# ✅ Fixed version of the predict_cluster function
def predict_cluster(kmeans, scaler, age, income, spending_score, gender_code=0, 
                    last_purchase=0, membership_years=1):
    try:
        input_data = {
            'age': age,
            'income': income,
            'spending_score': spending_score,
            'last_purchase_amount': last_purchase,
            'membership_years': membership_years,
            'purchase_frequency': 0
        }

        input_data['gender_Male'] = 1 if gender_code == 0 else 0
        input_data['gender_Female'] = 1 if gender_code == 1 else 0
        input_data['gender_Other'] = 1 if gender_code == 2 else 0

        preferred_categories = [
            'preferred_category_Electronics',
            'preferred_category_Groceries',
            'preferred_category_Home & Garden',
            'preferred_category_Sports'
        ]
        for cat in preferred_categories:
            input_data[cat] = 0

        if hasattr(kmeans, 'feature_names_in_'):
            feature_names = kmeans.feature_names_in_
        elif hasattr(scaler, 'feature_names_in_'):
            feature_names = scaler.feature_names_in_
        else:
            st.error("Model does not expose expected feature names.")
            return 0

        for feature in feature_names:
            if feature not in input_data:
                st.warning(f"Missing expected feature: {feature} - adding with default value 0")
                input_data[feature] = 0

        input_df = pd.DataFrame([input_data])[list(feature_names)]
        scaled_input = scaler.transform(input_df)
        cluster = kmeans.predict(scaled_input)[0]
        return cluster

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.error(f"Expected features: {feature_names if 'feature_names' in locals() else 'unknown'}")
        st.error(f"Provided features: {list(input_data.keys())}")
        return 0

def main():
    st.title("🧩 Smart Customer Segmentation Tool")
    st.markdown("### Enter customer information to identify their market segment")

    kmeans, scaler = load_models()
    if not kmeans or not scaler:
        st.error("Failed to load models. Please check if model files exist or if there are any errors.")
        return

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
        income = st.number_input("Annual Income ($)", 10000, 200000, 50000)
        spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)

    with col2:
        st.markdown("### Additional Details (Optional)")
        gender = st.radio("Gender", ["Male", "Female", "Other"], index=0)
        gender_code = 0 if gender == "Male" else 1 if gender == "Female" else 2
        last_purchase = st.number_input("Last Purchase Amount ($)", 0, 10000, int(income * 0.05))
        membership_years = st.number_input("Membership Years", 0, 20, 1)

    if st.button("Identify Customer Segment"):
        with st.spinner("Analyzing customer data..."):
            cluster = predict_cluster(kmeans, scaler, age, income, spending_score, gender_code, last_purchase, membership_years)

            cluster_info = {
                0: "💼 High Income - Moderate Spending",
                1: "🛍️ Low Income - High Spending",
                2: "💳 Moderate Income - Average Spending",
                3: "🔒 Low Income - Low Spending"
            }

            st.success(f"Customer belongs to Segment {cluster}")
            st.markdown(f"<div class='segment-box segment-{cluster}'>{cluster_info.get(cluster, 'Unknown Segment')}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
