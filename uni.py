# train_and_save.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier   # or LinearSVC
from sklearn import metrics
import joblib

# load cleaned data (adjust path)
df = pd.read_csv("data/clean.csv", encoding='latin1')  # or your file
# ensure columns exist
df["text"] = df["text"].fillna("").astype(str)
df["category"] = df["category"].astype(int)   # or label values used

X = df["text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# pipeline keeps vectorizer + model together
text_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,2))),
    ("clf", PassiveAggressiveClassifier(max_iter=1000, random_state=42))
])

# fit pipeline (this fits tfidf internally)
text_pipe.fit(X_train, y_train)

# evaluate (optional)
pred = text_pipe.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, pred))
print(metrics.classification_report(y_test, pred))

# save fitted pipeline
joblib.dump(text_pipe, "text_pipe.joblib")
print("Saved fitted pipeline -> text_pipe.joblib")
