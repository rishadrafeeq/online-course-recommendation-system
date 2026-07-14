# -*- coding: utf-8 -*-



import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
import os

# Function to load data (reusing the one from the notebook)
@st.cache_data
def load_data(file_path):
    """Load the dataset with optimized data types"""
    dtype = {
        'user_id': 'int32',
        'course_id': 'int32',
        'course_duration_hours': 'float32',
        'course_price': 'float32',
        'feedback_score': 'float32',
        'time_spent_hours': 'float32',
        'previous_courses_taken': 'int8',
        'rating': 'float32',
        'enrollment_numbers': 'int32'
    }
    df = pd.read_csv(data_path, dtype=dtype)
    return df

# Function to clean data (reusing the one from the notebook)
@st.cache_data
def clean_data(df):
    """Clean and preprocess the data"""
    df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0}).fillna(0)
    df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0}).fillna(0)
    difficulty_map = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
    df['difficulty_level'] = df['difficulty_level'].map(difficulty_map).fillna(-1) # Use -1 for unknown

    df = df.drop_duplicates()
    return df

# Function to train SVD model (reusing the one from the notebook)
@st.cache_resource # Cache the model itself
def train_svd_model(df):
    """Train SVD model on the full dataset"""
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    data = Dataset.load_from_df(df[['user_id', 'course_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)
    return svd, trainset # Return trainset to get inner IDs

# Function to get SVD recommendations (reusing the one from the notebook and adapting for Streamlit)
def svd_recommendations(model, trainset, df, user_id, n=5):
    """Generate recommendations using SVD"""
    try:
        user_inner_id = trainset.to_inner_uid(user_id)
    except ValueError:
        # Handle new users not in the training set by recommending popular courses
        st.warning(f"User ID {user_id} not found in training data. Recommending popular courses.")
        return get_popular_courses(df, n)

    # Get raw course IDs from the dataset
    all_course_ids = df['course_id'].unique()

    # Get courses user has already taken (using original df for simplicity)
    user_courses = df[df['user_id'] == user_id]['course_id'].unique()

    # Predict ratings for courses not taken
    predictions = []
    for course_id in all_course_ids:
        if course_id not in user_courses:
            # Use the raw IDs for prediction with the model trained on raw IDs
            pred = model.predict(user_id, course_id)
            predictions.append((course_id, pred.est))

    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top n recommendations
    top_course_ids = [x[0] for x in predictions[:n]]

    # Return course details
    return df[df['course_id'].isin(top_course_ids)][['course_id', 'course_name', 'instructor', 'rating']].drop_duplicates()

# Function to get popular courses (reusing the one from the notebook)
def get_popular_courses(df, n=5):
    """Get top n popular courses based on enrollment and rating"""
    popular = df.groupby(['course_id', 'course_name', 'instructor']).agg({
        'enrollment_numbers': 'sum',
        'rating': 'mean'
    }).reset_index()

    popular = popular.sort_values(['enrollment_numbers', 'rating'], ascending=False)
    return popular.head(n)[['course_id', 'course_name', 'instructor', 'rating']]


# --- Streamlit App ---
st.title("Online Course Recommendation System (SVD)")

# Load and clean data
data_path = "online_course_recommendation_v2.csv" # Adjust if needed
if not os.path.exists(data_path):
    st.error(f"Dataset not found at {data_path}. Please upload the file or update the path.")
else:
    df = load_data(data_path)
    df = clean_data(df)

    # Train the SVD model
    st.write("Training the SVD model...")
    svd_model, trainset = train_svd_model(df)
    st.write("Model training complete!")

    # User input for recommendations
    st.header("Get Course Recommendations")
    user_id_input = st.number_input("Enter User ID", min_value=int(df['user_id'].min()), max_value=int(df['user_id'].max()))
    num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)

    if st.button("Get Recommendations"):
        if user_id_input:
            recommendations = svd_recommendations(svd_model, trainset, df, user_id_input, num_recommendations)

            if not recommendations.empty:
                st.subheader(f"Recommendations for User ID {user_id_input}:")
                st.dataframe(recommendations)
            else:
                st.info("No recommendations found for this user.")
        else:
            st.warning("Please enter a User ID to get recommendations.")

    st.header("Explore Popular Courses")
    num_popular = st.slider("Number of popular courses to show", min_value=1, max_value=20, value=5)
    if st.button("Show Popular Courses"):
        popular_courses = get_popular_courses(df, num_popular)
        st.subheader(f"Top {num_popular} Popular Courses:")
        st.dataframe(popular_courses)