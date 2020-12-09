import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Read data
data = pd.read_csv("/home/raha/py/src/shovel/HW2/Q7/data/heart.csv")
# print(data.info())

y = data['target']
X = data.drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

X_train.loc[X_train['ca'] == 4, 'ca'] = np.NaN

# substitute nan with mode
X_train['ca'] = X_train['ca'].fillna(X_train['ca'].mode()[0])

X_train.loc[X_train['thal'] == 0, 'thal'] = np.NaN
X_train['thal'] = X_train['thal'].fillna(X_train['thal'].mode()[0])

duplicates = X_train.duplicated()
duplicated_index = X_train[duplicates].index
X_train.drop(duplicated_index, inplace=True)
y_train.drop(duplicated_index, inplace=True)

continous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for feature in continous_features:
    feature_data = X_train[feature]
    Q1 = np.percentile(feature_data, 25)
    Q3 = np.percentile(feature_data, 75)
    IQR = Q3 - Q1  # Interquartile Range
    outlier_step = IQR * 1.5  # considering above
    outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()
    X_train.drop(outliers, inplace=True)
    y_train.drop(outliers, inplace=True)
    print('For the feature {}, No of Outliers is {}'.format(feature, len(outliers)))
    print('Outliers from {} feature removed'.format(feature))

# columns = list(X_train)
# for i in range(len(columns)):
#     for j in range(i+1, len(columns)):
#         MI = mutual_info_score(X_train[columns[i]], X_train[columns[j]])
#         if MI > 1:
#             print(columns[i], columns[j], MI)

# print(mutual_info_score(X_train[X_train], X_train[1]))

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

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_predicted = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_predicted)
print(accuracy)
