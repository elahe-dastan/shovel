import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/student.csv")

X = df.iloc[:, :-1]
Y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, Y)
