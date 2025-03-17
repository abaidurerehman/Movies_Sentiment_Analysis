 Movie Review Sentiment Analysis Report
1 Project Overview
This project performs sentiment analysis on movie reviews using machine learning. The goal is to classify reviews as positive or negative based on their sentiment.

2 Approach Used
Data Preprocessing

Converted text to lowercase
Removed punctuation, stopwords, and numbers
Performed tokenization
Applied TF-IDF vectorization
Model Selection

Trained multiple models:
Logistic Regression  (Best Performance)
Naïve Bayes
Support Vector Machine (SVM) (Slow but effective)
Evaluated models using accuracy and F1-score
Deployment

Built an interactive Streamlit web app
Users can enter a review, and the model predicts its sentiment
Hosted on localhost (configurable)
3 Challenges Faced
SVM was slow: Took too long to train and predict. Switched to Logistic Regression.
Large Dataset Processing: TF-IDF transformation was memory-intensive.  Used n-grams to optimize.
Stopword Removal Impact: Some models performed worse with stopword removal.  Adjusted preprocessing accordingly.
4 Model Performance
Model	Accuracy	F1-Score
Logistic Regression	89.2%	88.5%
Naïve Bayes	84.6%	83.2%
SVM (Linear)	90.1%	89.8% (but slow)
Best Model:  Logistic Regression (Good accuracy + fast inference)
