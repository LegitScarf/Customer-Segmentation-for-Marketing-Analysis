# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 18:04:58 2025

@author: KIIT
"""
import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🧩",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
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
    .segment-0 {
        background-color: #d4f1f9;
        color: #05445E;
    }
    .segment-1 {
        background-color: #ffecdb;
        color: #e67e22;
    }
    .segment-2 {
        background-color: #e8f4ea;
        color: #27ae60;
    }
    .segment-3 {
        background-color: #f2e4ff;
        color: #8e44ad;
    }
</style>
""", unsafe_allow_html=True)

# Load the model files
@st.cache_resource
def load_models():
    try:
        # Check if models exist in the local directory
        if os.path.exists('kmeans_model.pkl') and os.path.exists('scaler.pkl'):
            with open('kmeans_model.pkl', 'rb') as f:
                kmeans = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            return kmeans, scaler
        
        # If models don't exist locally, create simple versions
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        st.warning("Model files not found. Creating sample models for demonstration...")
        
        # Create a simple KMeans model with 4 clusters
        kmeans = KMeans(n_clusters=4, random_state=42)
        scaler = StandardScaler()
        
        # Fit with some dummy data to initialize
        dummy_data = np.random.rand(100, 3) * [100, 200000, 100]  # Age, Income, Spending
        dummy_df = pd.DataFrame(dummy_data, columns=['age', 'income', 'spending_score'])
        
        # Scale the data
        scaled_data = scaler.fit_transform(dummy_df)
        
        # Fit the KMeans model
        kmeans.fit(scaled_data)
        
        # Save the models
        with open('kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        return kmeans, scaler
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Predict cluster using customer data
def predict_cluster(kmeans, scaler, age, income, spending_score, gender_code=0, 
                    last_purchase=0, membership_years=1):
    try:
        # Create DataFrame with input data - start with basic features
        input_data = {
            'age': age,
            'income': income,
            'spending_score': spending_score
        }
        
        # Get feature names expected by the model if available
        if hasattr(kmeans, 'feature_names_in_'):
            feature_names = kmeans.feature_names_in_
        elif hasattr(scaler, 'feature_names_in_'):
            feature_names = scaler.feature_names_in_
        else:
            # Default to basic features if not available
            feature_names = ['age', 'income', 'spending_score']
        
        # Add additional features if they're expected
        if 'gender' in feature_names:
            input_data['gender'] = gender_code
        
        if 'gender_Male' in feature_names:
            input_data['gender_Male'] = 1 if gender_code == 0 else 0
            input_data['gender_Female'] = 1 if gender_code == 1 else 0
            input_data['gender_Other'] = 1 if gender_code == 2 else 0
            
        if 'last_purchase_amount' in feature_names:
            input_data['last_purchase_amount'] = last_purchase
            
        if 'membership_years' in feature_names:
            input_data['membership_years'] = membership_years
        
        # Add preferred category features if needed
        categories = ['Electronics', 'Groceries', 'Home & Garden', 'Sports']
        for cat in categories:
            column_name = f'preferred_category_{cat.replace(" & ", "_")}'
            if column_name in feature_names:
                input_data[column_name] = 0  # Set default to 0
        
        # Create DataFrame with our input data
        input_df = pd.DataFrame([input_data])
        
        # Check if we're missing any expected features
        for feature in feature_names:
            if feature not in input_df.columns:
                st.warning(f"Missing expected feature: {feature} - adding with default value 0")
                input_df[feature] = 0
        
        # Make sure the input dataframe has exactly the columns needed by the model
        input_df = input_df[feature_names]
        
        # Scale the input data
        scaled_input = scaler.transform(input_df)
        
        # Predict cluster
        cluster = kmeans.predict(scaled_input)[0]
        
        return cluster
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.error(f"Expected features: {feature_names if 'feature_names' in locals() else 'unknown'}")
        st.error(f"Provided features: {list(input_data.keys())}")
        return 0  # Return default cluster on error

# Main Streamlit app
def main():
    st.title("🧩 Smart Customer Segmentation Tool")
    st.markdown("### Enter customer information to identify their market segment")
    
    # Load models
    kmeans, scaler = load_models()
    
    if not kmeans or not scaler:
        st.error("Failed to load models. Please check if model files exist or if there are any errors.")
        return
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # Input fields
    with col1:
        age = st.slider("Age", 18, 100, 30)
        income = st.number_input("Annual Income ($)", 10000, 200000, 50000)
        spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)
    
    # Additional fields
    with col2:
        st.markdown("### Additional Details (Optional)")
        gender = st.radio("Gender", ["Male", "Female", "Other"], index=0)
        gender_code = 0 if gender == "Male" else 1 if gender == "Female" else 2
        
        last_purchase = st.number_input("Last Purchase Amount ($)", 0, 10000, int(income * 0.05))
        membership_years = st.number_input("Membership Years", 0, 20, 1)
        
        # Add category selection if needed
        if hasattr(kmeans, 'feature_names_in_') and any('preferred_category' in feature for feature in kmeans.feature_names_in_):
            st.markdown("### Preferred Shopping Category")
            category = st.selectbox("Category", ["Electronics", "Groceries", "Home & Garden", "Sports"])
    
    # Prediction button
    if st.button("Identify Customer Segment"):
        with st.spinner("Analyzing customer data..."):
            try:
                cluster = predict_cluster(
                    kmeans, scaler, age, income, spending_score, 
                    gender_code, last_purchase, membership_years
                )
                
                # Cluster descriptions
                cluster_info = {
                    0: "💼 **High Income - Moderate Spending**\n\nThese customers have significant purchasing power but are selective about their spending. They prefer quality over quantity and respond well to premium offerings and loyalty benefits.",
                    1: "🛍️ **Low Income - High Spending**\n\nThese customers spend beyond their means. They're likely responsive to financing options, discounts, and may be emotional shoppers. Focus on affordability messaging.",
                    2: "💳 **Moderate Income - Average Spending**\n\nThe balanced segment. These customers are your typical middle-market consumers who make practical purchasing decisions. They appreciate value and consistent quality.",
                    3: "🔒 **Low Income - Low Spending**\n\nThese cost-conscious customers prioritize essentials and hunt for the best deals. They respond well to promotions, discounts, and budget offerings."
                }
                
                # Display results
                st.success(f"Customer belongs to Segment {cluster}")
                
                # Show segment details
                st.markdown(f"<div class='segment-box segment-{cluster}'>{cluster_info.get(cluster, 'No segment information available')}</div>", unsafe_allow_html=True)
                
                # Create visualization tabs
                st.markdown("### Customer Profile Analysis")
                tab1, tab2 = st.tabs(["Radar Chart", "Customer Comparisons"])
                
                with tab1:
                    # Create radar chart
                    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
                    
                    # Categories and values
                    categories = ['Age', 'Income', 'Spending Score']
                    
                    # Normalize values for radar chart
                    norm_age = age / 100
                    norm_income = income / 200000
                    norm_spending = spending_score / 100
                    
                    values = [norm_age, norm_income, norm_spending]
                    
                    # Plot setup
                    N = len(categories)
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # Close the loop
                    values += values[:1]  # Close the loop
                    
                    # Draw the plot
                    ax.plot(angles, values, 'o-', linewidth=2)
                    ax.fill(angles, values, alpha=0.25)
                    
                    # Set category labels
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    
                    # Remove y-axis labels
                    ax.set_yticklabels([])
                    
                    # Add title
                    plt.title(f"Customer Profile - Segment {cluster}")
                    
                    # Show the graph
                    st.pyplot(fig)
                
                with tab2:
                    # Create comparison chart
                    st.markdown("#### How this customer compares to segment averages")
                    
                    # Sample segment averages (replace with your actual segment data)
                    segment_avgs = {
                        0: [45, 150000, 60],  # High Income - Moderate Spending
                        1: [30, 40000, 85],   # Low Income - High Spending
                        2: [40, 80000, 50],   # Moderate Income - Average Spending
                        3: [55, 35000, 25]    # Low Income - Low Spending
                    }
                    
                    # Get the averages for the predicted cluster
                    avg_age, avg_income, avg_spending = segment_avgs[cluster]
                    
                    # Create bar chart to compare customer with segment average
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Data
                    features = ['Age', 'Income ($K)', 'Spending Score']
                    customer_data = [age, income/1000, spending_score]
                    segment_data = [avg_age, avg_income/1000, avg_spending]
                    
                    # Set position of bars on X axis
                    x = np.arange(len(features))
                    width = 0.35
                    
                    # Create bars
                    customer_bars = ax.bar(x - width/2, customer_data, width, label='This Customer')
                    segment_bars = ax.bar(x + width/2, segment_data, width, label=f'Segment {cluster} Average')
                    
                    # Add labels and title
                    ax.set_ylabel('Value')
                    ax.set_title('Customer vs Segment Average')
                    ax.set_xticks(x)
                    ax.set_xticklabels(features)
                    ax.legend()
                    
                    # Add value labels on bars
                    def add_labels(bars):
                        for bar in bars:
                            height = bar.get_height()
                            ax.annotate(f'{height:.0f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),  # 3 points vertical offset
                                        textcoords="offset points",
                                        ha='center', va='bottom')
                    
                    add_labels(customer_bars)
                    add_labels(segment_bars)
                    
                    # Show plot
                    st.pyplot(fig)
                
                # Add marketing recommendations
                st.markdown("### 📊 Marketing Recommendations")
                
                marketing_recommendations = {
                    0: [
                        "Focus on premium product offerings",
                        "Emphasize quality and exclusivity in messaging",
                        "Offer loyalty programs with unique experiences",
                        "Use sophisticated, elegant branding"
                    ],
                    1: [
                        "Provide financing options and payment plans",
                        "Create flash sales and limited-time offers",
                        "Emphasize aspirational messaging",
                        "Offer rewards for social sharing"
                    ],
                    2: [
                        "Highlight value proposition and practical benefits",
                        "Create mid-tier product offerings",
                        "Use testimonials emphasizing reliability",
                        "Develop reasonable loyalty discounts"
                    ],
                    3: [
                        "Focus on essential features and affordability",
                        "Offer aggressive discounts and promotions",
                        "Emphasize cost savings in messaging",
                        "Create budget-friendly product lines"
                    ]
                }
                
                recommendations = marketing_recommendations.get(cluster, [])
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("There might be a mismatch between your model's expected features and what you're providing.")

# Run the app
if __name__ == "__main__":
    main()
