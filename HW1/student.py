import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/student.csv")
print(df)

binary_attributes = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
categorical_attributes = ["Mjob", "Fjob", "reason", "guardian"]
preprocess_pipeline = ColumnTransformer([
    ("binary", OrdinalEncoder(), binary_attributes),
    ("categorical", OneHotEncoder(), categorical_attributes)
])

data = preprocess_pipeline.fit_transform(df)
print(type(data))

X = df.iloc[:, :-1]
Y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, Y)
#
# lr = LinearRegression()
# lr.fit(X_train, y_train)
