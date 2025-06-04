import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer

# Load models with relative paths
with open('nb_classifier.pkl', 'rb') as classifier_file:
    nb_classifier = pickle.load(classifier_file)

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    count_vectorizer = pickle.load(vectorizer_file)

# Function to clean the input text
def process_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)                 # Remove URLs
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)          # Remove @mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)             # Remove special characters
    return text

# Sentiment label mapping
sentiment_mapping = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

# Main Streamlit app
def main():
    col1, col2, col3, col4 = st.columns([1, 1, 3, 1])
    with col3:
        st.image('download.png', width=200)
        st.title("Sentiment Analysis App")
        st.write("Enter a Twitter text below:")
        input_text = st.text_area("Input Text:", "")
        
        if st.button("Analyze"):
            cleaned_text = process_text(input_text)
            vectorized_text = count_vectorizer.transform([cleaned_text])
            sentiment_prediction = nb_classifier.predict(vectorized_text)[0]
            predicted_sentiment = sentiment_mapping.get(sentiment_prediction, "Unknown")

            st.subheader("Predicted Sentiment:")
            st.success(predicted_sentiment)

if __name__ == "__main__":
    main()
