import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/student.csv")

preprocess_pipeline =

X = df.iloc[:, :-1]
Y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, Y)

lr = LinearRegression()
lr.fit(X_train, y_train)
