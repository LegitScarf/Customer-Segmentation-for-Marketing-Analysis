# Modified version of app2 (1).py with corrected predict_cluster()
# NOTE: This is a patch to fix feature mismatches and make clustering predictions work reliably

import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🧩",
    layout="wide"
)

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

# ✅ Corrected Function
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

        # One-hot encode gender
        input_data['gender_Male'] = 1 if gender_code == 0 else 0
        input_data['gender_Female'] = 1 if gender_code == 1 else 0
        input_data['gender_Other'] = 1 if gender_code == 2 else 0

        # Add preferred category columns
        for cat in ['preferred_category_Electronics', 'preferred_category_Groceries',
                    'preferred_category_Home & Garden', 'preferred_category_Sports']:
            input_data[cat] = 0

        # Fetch model's expected features
        if hasattr(kmeans, 'feature_names_in_'):
            feature_names = kmeans.feature_names_in_
        elif hasattr(scaler, 'feature_names_in_'):
            feature_names = scaler.feature_names_in_
        else:
            st.error("Expected features not found in model.")
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
