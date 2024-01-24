import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

a = pd.read_csv("we.csv")
df = pd.DataFrame(a)
df = df[:len(df) - 1]
b = pd.read_csv("data1.csv")
df1 = pd.DataFrame(b)
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df1['Label'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
y_train = y_train.values.ravel()
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
joblib.dump((model, vectorizer), 'trained_model_with_vectorizer.joblib')
a=vectorizer.transform(["We would like to send you awesome offers!Notifications can be turned off anytime from settings."
])
y_pred = model.predict(a)
print(y_pred)


