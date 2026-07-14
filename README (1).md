🎓 Online Course Recommendation System
A collaborative filtering-based recommendation system built with SVD (Singular Value Decomposition) and deployed via Streamlit that provides personalized course recommendations to users based on their learning history and preferences.

📊 Project Overview
This system analyzes user-course interactions and ratings to predict which courses a user would most likely enjoy. It handles both existing users (using SVD collaborative filtering) and new users (via popularity-based fallback recommendations).

🚀 Key Features
Personalized Recommendations: SVD-based collaborative filtering for accurate predictions

Cold Start Handling: Popularity-based recommendations for new users

Interactive Web Interface: User-friendly Streamlit dashboard

Real-time Predictions: Instant recommendations based on user input

Data Optimization: Efficient data processing with optimized data types

🛠️ Tech Stack
Python (Pandas, NumPy)

Surprise (Scikit-learn for recommender systems)

Streamlit (Web deployment)

Machine Learning: SVD algorithm for matrix factorization

📈 How It Works
Data Processing: Cleans and preprocesses course data with categorical encoding

Model Training: Trains SVD model on user-course rating matrix

Prediction: Estimates user ratings for unseen courses

Recommendation: Returns top-N courses with highest predicted ratings

📁 Dataset Features
User IDs and Course IDs

Course metadata (duration, price, difficulty level)

User interactions (time spent, feedback scores)

Ratings and enrollment numbers

Certification and study material availability

🎯 Usage
Enter a User ID in the web interface

Select number of recommendations desired

Get instant personalized course suggestions

Explore popular courses as an alternative
