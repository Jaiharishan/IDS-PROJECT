import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta



# Set page configuration
st.set_page_config(
   page_title="Order Delivery Prediction",
   page_icon="🚚",
   layout="wide",
   initial_sidebar_state="expanded",
)


# Custom CSS for styling
st.markdown("""
<style>
   .time-card {
       border-radius: 10px;
       padding: 20px;
       margin: 10px 0;
       box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
       transition: 0.3s;
   }
   .time-card:hover {
       box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
   }
   .days {
       background-color: #262730;
       border-left: 5px solid #4e79a7;
   }
   .hours {
       background-color: #262730;
       border-left: 5px solid #e15759;
   }
   .big-font {
       font-size: 2.5rem !important;
       font-weight: 700;
       color: #f0f0f0;
   }
   .header-font {
       font-size: 1.2rem !important;
       font-weight: 600;
       color: #f0f0f0;
   }
</style>
""", unsafe_allow_html=True)


# Load Data
df = pd.read_csv('data/full_df2.csv')


# Set title and description
st.title("E-Commerce Prediction App")


st.markdown(
   """
   This app predicts the delivery time of orders based on various features.
   It also provides visualizations and sentiment analysis of reviews.
   """
)


# Replace the sidebar navigation with tabs
tabs = st.tabs(["Delivery Times Prediction", "Freight Value Prediction", "Visualization", "NLP Reviews Analysis"])


with tabs[0]:
    # The prediction code is already in place below
    st.title("Predicting Order Delivery Times 🚚")


    # Load the Model
    model = load('./Models/random_forest_model.pkl')


    # Get user inputs
    features = ['product_category_name', 'customer_city', 'freight_value', 'delivery_difference', 'order_purchase_month']


    # Style the inputs
    st.markdown("### Input Features")
    col1, col2 = st.columns(2)


    with col1:
        product_category_name = st.selectbox('Product Category Name', df['product_category_name'].unique())
        customer_city = st.selectbox('Customer City', df['customer_city'].unique())


    with col2:
        freight_value = st.number_input('Freight Value', min_value=0.0, step=0.1)
        delivery_difference = st.number_input('Delivery Difference', min_value=0.0, step=0.1)


    order_purchase_month = st.selectbox(
        'Order Purchase Month',
        {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }.keys()
    )


    # Convert the selected month to its corresponding numeric value
    order_purchase_month = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }[order_purchase_month]

    # Encode the categorical features
    # Initialize LabelEncoders
    product_category_encoder = LabelEncoder()
    customer_city_encoder = LabelEncoder()
    order_purchase_month_encoder = LabelEncoder()

    # Fit the encoders on the respective columns
    df['product_category_name_encoded'] = product_category_encoder.fit_transform(df['product_category_name'])
    df['customer_city_encoded'] = customer_city_encoder.fit_transform(df['customer_city'])
    df['order_purchase_month_encoded'] = order_purchase_month_encoder.fit_transform(df['order_purchase_month'])


    # Transform user inputs
    product_category_name = product_category_encoder.transform([product_category_name])[0]
    customer_city = customer_city_encoder.transform([customer_city])[0]
    order_purchase_month = order_purchase_month_encoder.transform([order_purchase_month])[0]


    # Make predictions
    if st.button('Predict'):
        input_data = pd.DataFrame({
            'product_category_name_encoded': [product_category_name],
            'customer_city_encoded': [customer_city],
            'order_purchase_month_encoded': [order_purchase_month],
            'freight_value': [freight_value],
            'delivery_difference': [delivery_difference],
        })


        prediction = model.predict(input_data)
        days = int(prediction[0])
        hours = int((prediction[0] - days) * 24)


        col1, col2 = st.columns([1, 1])




        with col1:
            st.markdown(f"""
            <div class="time-card days">
                <div class="header-font">Days</div>
                <div class="big-font">{days}</div>
            </div>
            """, unsafe_allow_html=True)
        


        with col2:
            st.markdown(f"""
            <div class="time-card hours">
                <div class="header-font">Hours</div>
                <div class="big-font">{hours}</div>
            </div>
            """, unsafe_allow_html=True)
    
    
    
with tabs[1]:   
    # Prediction for Freight Value
    st.title("Predicting Item Freight Values 💸")

    # Load the Model
    freight_model = load('./Models/rf_freight_model.pkl')

    # Get user inputs
    freight_features = ['product_weight_g', 'total_order_value', 'price']

    # Style the inputs
    st.markdown("### Input Features")
    col1, col2 = st.columns(2)

    with col1:
        product_weight_g = st.number_input('Product Weight in Grams',  min_value=0.0, max_value=10000.0, step=0.1)
        total_order_value = st.number_input('Total Order Value', min_value=0.0, max_value=100000.0, step=0.1)

    with col2:
        price = st.number_input('Price', min_value=0.0, max_value=10000.0, step=0.1)

    # Make predictions
    if st.button('Predict Freight'):
        # Create input DataFrame without encoding
        freight_input_data = pd.DataFrame({
            'product_weight_g': [product_weight_g],
            'total_order_value': [total_order_value],
            'price': [price]
        })

        # Make prediction
        freight_prediction = freight_model.predict(freight_input_data)
        freight = round(float(freight_prediction[0]), 2)  # Round to 2 decimal places

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"""
            <div class="time-card days">
                <div class="header-font">Freight Value (R$)</div>
                <div class="big-font">{freight}</div>
            </div>
            """, unsafe_allow_html=True)
            



with tabs[2]:
    st.markdown("## Visualization Section")


    # Create a section for displaying correlations
    st.markdown("### Correlation Between Columns")
    if st.checkbox('Show Correlation Matrix'):
        numeric_columns = df.select_dtypes(include=['number'])
        correlation_matrix = numeric_columns.corr()
        st.write(correlation_matrix)
        st.markdown("#### Heatmap of Correlation Matrix")
        # Display the heatmap
        fig, ax = plt.subplots(figsize=(12, 12))
        
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, annot_kws={'size': 7},  square=True)
        plt.xticks(rotation=45, ha='right', fontsize=6)  # Reduce x-axis label size
        plt.yticks(rotation=0, fontsize=6)  # Reduce y-axis label size
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        st.pyplot(fig)


with tabs[3]:
    st.markdown("## NLP Reviews Analysis Section")
    # Add your NLP analysis code here
    # Load the SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()


    # Get user input for review
    st.markdown("### Enter a Review")
    user_review = st.text_area("Type your review here:")


    if st.button("Analyze Sentiment"):
        if user_review.strip():
            # Get sentiment scores
            sentiment_scores = sia.polarity_scores(user_review)
            st.markdown("### Sentiment Analysis Results")


            if sentiment_scores['compound'] > 0:
                st.markdown("### Overall Sentiment: **Positive**")
                st.success("Positive Sentiment")
            elif sentiment_scores['compound'] < 0:
                st.markdown("### Overall Sentiment: **Negative**")
                st.error("Negative Sentiment")
            else:
                st.markdown("### Overall Sentiment: **Neutral**")
                st.warning("Neutral Sentiment")

