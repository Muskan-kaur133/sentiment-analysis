import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model & vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and find out whether it's **Positive** or **Negative**.")

user_input = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")
