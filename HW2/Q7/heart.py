import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Read data
data = pd.read_csv("./data/heart.csv")
# print(data.info())

y = data['target']
X = data.drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Do not look at the test data
standardize = ColumnTransformer([
    ("standardize", StandardScaler(), list(X_train))
])

# fit and transform
normalized_train_data = standardize.fit_transform(X_train)
# just transform
transformed_test_data = standardize.transform(X_test)

knn_clf = KNeighborsClassifier()
knn_clf.fit(normalized_train_data, y_train)

y_predicted = knn_clf.predict(transformed_test_data)

accuracy = accuracy_score(y_test, y_predicted)
print(accuracy)

