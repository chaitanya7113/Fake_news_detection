from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
import pandas as pd

df = pd.read_csv("data/clean.csv")  # Load your dataset
X = df["text"]  # feature
y = df["category"]  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

text_clf = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])
text_clf.fit(X_train, y_train)
predictions = text_clf.predict(X_test)
print(metrics.classification_report(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))