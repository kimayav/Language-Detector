import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv('/languageDataset.csv')
X = data["Text"]
y = data["Language"]
le = LabelEncoder() 
y = le.fit_transform(y)
data_list = []
for text in X:

        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower()

        data_list.append(text)
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
X.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

def predict(text):
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    return lang[0]
inp=input("ENTER THE SENTENCE:")
predict(inp)

