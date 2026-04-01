import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model & vectorizer (UPDATED FILE NAMES)
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

STOPWORDS = set(stopwords.words('english'))

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # improved
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Sentiment Analysis App")

st.title("🎬 Movie Review Sentiment Analysis")

st.write("Analyze single or multiple reviews at once!")

# ----------------------------
# SINGLE REVIEW
# ----------------------------
st.subheader("🔹 Single Review Analysis")

single_input = st.text_area("Enter a single review:")

if st.button("Analyze Single Review"):
    if single_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(single_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

# ----------------------------
# MULTIPLE REVIEWS
# ----------------------------
st.subheader("🔹 Multiple Reviews Analysis")

multi_input = st.text_area(
    "Enter multiple reviews (one per line):",
    height=200
)

if st.button("Analyze Multiple Reviews"):
    if multi_input.strip() == "":
        st.warning("Please enter some reviews.")
    else:
        # Split and clean input
        reviews = [r.strip() for r in multi_input.split("\n") if r.strip() != ""]

        results = []

        for review in reviews:
            cleaned = clean_text(review)
            vector = tfidf.transform([cleaned])   # one review at a time
            prediction = model.predict(vector)[0]

            sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

            results.append({
                "Review": review,
                "Sentiment": sentiment
            })

        df = pd.DataFrame(results)

        st.write("### Results")
        st.dataframe(df)
        
        
# import streamlit as st
# import joblib
# import re
# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords')

# # Load model & vectorizer
# model = joblib.load("model.pkl")
# tfidf = joblib.load("tfidf_vectorizer.pkl")

# # Text cleaning function
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'[^a-zA-Z]', ' ', text)
#     words = text.split()
#     words = [w for w in words if w not in stopwords.words('english')]
#     return ' '.join(words)

# # Streamlit UI
# st.set_page_config(page_title="Sentiment Analysis App")

# st.title("🎬 Movie Review Sentiment Analysis")
# st.write("Enter a movie review and find out whether it's **Positive** or **Negative**.")

# user_input = st.text_area("Enter your review here:")

# if st.button("Analyze Sentiment"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         cleaned = clean_text(user_input)
#         vectorized = tfidf.transform([cleaned])
#         prediction = model.predict(vectorized)[0]

#         if prediction == 1:
#             st.success("😊 Positive Review")
#         else:
#             st.error("😞 Negative Review")
