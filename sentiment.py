import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load model & vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Download stopwords (only once)
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Function to predict sentiment
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    transformed_review = vectorizer.transform([processed_review])
    prediction = model.predict(transformed_review)[0]
    return "😊 Positive" if prediction == 1 else "😠 Negative"
