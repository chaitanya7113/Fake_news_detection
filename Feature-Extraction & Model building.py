from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("data/clean.csv")

X = df["text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=5,
        max_df=0.8
    )),
    ("clf", LinearSVC())
])

text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predictions))
print("Accuracy:", metrics.accuracy_score(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))

# SAVE WITH COMPRESSION
joblib.dump(text_clf, "text_model.joblib", compress=9)
