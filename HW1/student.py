import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/student.csv")
print(df['school'])
# X_train, X_test, y_train, y_test = train_test_split()