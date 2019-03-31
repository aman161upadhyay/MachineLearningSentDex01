import numpy as np
import pandas as pd
import quandl
import math
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

quandl.ApiConfig.api_key = 'KRNbFEUBQY4sDcQKhGGe'
df = quandl.get('WIKI/GOOGL')
print(df.head())

# Got errors because the latest version of numpy was broken(again!), So installed an older version of it.

df = df[['Adj. Open', 'Adj. Close', 'Adj. High', 'Adj. Low', 'Adj. Volume']]
df['Volatility'] = ((df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']) * 100
df['percent_change'] = ((df['Adj. Open'] - df['Adj. Close']) / df['Adj. Close']) * 100
df = df[['Adj. Close', 'Volatility', 'percent_change', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)

print(len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = LinearRegression()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print("Accuracy of Linear Regression is:", accuracy)

# classifier = svm.SVR()                  To show how easy it is to switch between different ML algorithms.
# classifier.fit(X_train, y_train)
# accuracy = classifier.score(X_test, y_test)
# print("Accuracy of SVM is:", accuracy)

# classifier = svm.SVR(kernel='poly')         Another example.




