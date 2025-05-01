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

# Function to inspect model features
def inspect_model_features(kmeans, scaler):
    try:
        # Get feature names from scaler or kmeans model
        if hasattr(scaler, 'feature_names_in_'):
            feature_names = scaler.feature_names_in_
            st.sidebar.write("### Expected Features (from scaler)")
            st.sidebar.write(", ".join(feature_names))
            return feature_names
        elif hasattr(kmeans, 'feature_names_in_'):
            feature_names = kmeans.feature_names_in_
            st.sidebar.write("### Expected Features (from kmeans)")
            st.sidebar.write(", ".join(feature_names))
            return feature_names
        else:
            st.sidebar.warning("Model doesn't have feature_names_in_ attribute")
            return ['age', 'income', 'spending_score']
            
    except Exception as e:
        st.sidebar.error(f"Error inspecting model features: {str(e)}")
        return ['age', 'income', 'spending_score']  # Return default features

# Predict cluster using customer data
def predict_cluster(kmeans, scaler, feature_names, user_inputs):
    try:
        # Create a DataFrame with exactly the expected columns
        input_df = pd.DataFrame(columns=feature_names)
        
        # Fill with zeros first (baseline)
        input_df.loc[0] = [0] * len(feature_names)
        
        # Now update with the actual values we have
        for feature in feature_names:
            if feature in user_inputs:
                input_df.loc[0, feature] = user_inputs[feature]
        
        # Special handling for purchase_frequency if missing but required
        if 'purchase_frequency' in feature_names and 'purchase_frequency' not in user_inputs:
            # Derive from other inputs if possible
            if 'last_purchase_amount' in user_inputs and 'income' in user_inputs:
                # Simple heuristic: higher spending relative to income suggests higher frequency
                ratio = user_inputs.get('last_purchase_amount', 0) / max(user_inputs.get('income', 1), 1)
                input_df.loc[0, 'purchase_frequency'] = min(int(ratio * 20), 10)  # Scale to reasonable value
        
        # Scale the input data
        scaled_input = scaler.transform(input_df)
        
        # Predict cluster
        cluster = kmeans.predict(scaled_input)[0]
        
        return cluster
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.error(f"Expected features: {feature_names}")
        st.error(f"Provided features: {list(user_inputs.keys())}")
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
    
    # Inspect model features
    feature_names = inspect_model_features(kmeans, scaler)
    
    # Debug option in sidebar
    st.sidebar.title("Debug Options")
    show_debug = st.sidebar.checkbox("Show Debug Information")
    if show_debug:
        st.sidebar.write("### Model Information")
        st.sidebar.write(f"KMeans n_clusters: {kmeans.n_clusters}")
        
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # Track all user inputs in a dictionary
    user_inputs = {}
    
    # Input fields
    with col1:
        user_inputs['age'] = st.slider("Age", 18, 100, 30)
        user_inputs['income'] = st.number_input("Annual Income ($)", 10000, 200000, 50000)
        user_inputs['spending_score'] = st.slider("Spending Score (1-100)", 1, 100, 50)
    
    # Additional fields
    with col2:
        st.markdown("### Additional Details")
        # Handle gender column exactly as expected by the model
        if 'gender' in feature_names:
            gender_options = ["Male", "Female", "Other"]
            gender_index = st.radio("Gender", gender_options, index=0)
            user_inputs['gender'] = gender_options.index(gender_index)
        
        # Purchase history
        user_inputs['last_purchase_amount'] = st.number_input("Last Purchase Amount ($)", 0, 10000, int(user_inputs['income'] * 0.05))
        user_inputs['membership_years'] = st.number_input("Membership Years", 0, 20, 1)
        
        if 'purchase_frequency' in feature_names:
            user_inputs['purchase_frequency'] = st.slider("Purchase Frequency (per month)", 0, 10, 2)
    
    # Preferred categories section
    if any('preferred_category' in feature for feature in feature_names):
        st.markdown("### Preferred Shopping Category")
        categories = ["Electronics", "Groceries", "Home & Garden", "Sports"]
        selected_category = st.selectbox("Primary Category", categories)
        
        # Set all category flags correctly
        for category in categories:
            category_feature = f"preferred_category_{category.replace(' & ', '_')}"
            if category_feature in feature_names:
                user_inputs[category_feature] = 1 if category == selected_category else 0
    
    # For debugging
    if show_debug:
        st.sidebar.write("### User Inputs")
        st.sidebar.write(user_inputs)
    
    # Prediction button
    if st.button("Identify Customer Segment"):
        with st.spinner("Analyzing customer data..."):
            try:
                cluster = predict_cluster(kmeans, scaler, feature_names, user_inputs)
                
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
                    norm_age = user_inputs['age'] / 100
                    norm_income = user_inputs['income'] / 200000
                    norm_spending = user_inputs['spending_score'] / 100
                    
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
                    customer_data = [user_inputs['age'], user_inputs['income']/1000, user_inputs['spending_score']]
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
                if show_debug:
                    import traceback
                    st.error(traceback.format_exc())

# Run the app
if __name__ == "__main__":
    main()
