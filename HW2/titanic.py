import pandas as pd
import re
import sys
from sklearn.tree import DecisionTreeClassifier

# Read data
train_data = pd.read_csv("data/train.csv")
# print(train_data.info())

# Fix null values

# null age --> median
age = train_data['Age']
age.fillna(age.median, inplace=True)

# null embarked --> mode
embarked = train_data['Embarked']
embarked.fillna(embarked.mode()[0], inplace=True)

# drop cabin
train_data = train_data.drop('Cabin', axis=1)

# drop passenger id
train_data = train_data.drop('PassengerId', axis=1)
# print(train_data.info())

# extracting honorifics
names = train_data['Name']
honorifics = []
for name in names:
    start = name.find(',') + 1
    end = name.find('.')
    honorifics.append(name[start:end])

# add honorific column
train_data['Honorific'] = honorifics

sys.setrecursionlimit(len(set(honorifics)) * len(honorifics) * 2)

# find age average for each honorific
for honorific in set(honorifics):
    print(honorific)
    for index, row in train_data.iterrows():
        if row['Honorific'] == honorific:
            print(row['Honorific'], row['Age'])

Y = train_data['Survived']
X = train_data.drop('Survived', axis=1)

# tree = DecisionTreeClassifier(max_depth=5)
# tree.fit(X, Y)
