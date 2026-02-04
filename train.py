import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords (only checks, fast)
nltk.download('stopwords')

# Load stopwords ONCE
STOPWORDS = set(stopwords.words('english'))

# ----------------------------
# Text preprocessing function
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)

print("Loading dataset...")
df = pd.read_csv("IMDB Dataset.csv")

# ⚠️ LIMIT DATA FOR SPEED (VERY IMPORTANT)
df = df.sample(20000, random_state=42)  # use 20k reviews

print("Cleaning text...")
df['review'] = df['review'].apply(clean_text)

print("Encoding labels...")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'],
    test_size=0.2, random_state=42
)

print("Vectorizing text...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("Saving model...")
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("✅ Training completed successfully!")

