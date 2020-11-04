import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/student.csv")

binary_attributes = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
categorical_attributes = ["Mjob", "Fjob", "reason", "guardian"]
numerical_attributes = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
preprocess_pipeline = ColumnTransformer([
    ("binary", OrdinalEncoder(), binary_attributes),
    ("categorical", OneHotEncoder(), categorical_attributes)
])

preprocessed_cat_df = preprocess_pipeline.fit_transform(df)
num_columns = df[numerical_attributes].to_numpy()
preprocessed_df = np.append(preprocessed_cat_df, num_columns, axis=1)

X = preprocessed_df[:, :-1]
Y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, Y)

lr = LinearRegression()
lr.fit(X_train, y_train)

predicted = lr.predict(X_test)

lin_mse = mean_squared_error(y_test, predicted)
print(lin_mse)
