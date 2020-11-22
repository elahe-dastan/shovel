import pandas as pd
import re
import sys
from sklearn.tree import DecisionTreeClassifier

# Read data
train_data = pd.read_csv("data/train.csv")
# print(train_data.info())

# Fix null values

# null age --> median
# age = train_data['Age']
# age.fillna(age.median(), inplace=True)

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

# find age average for each honorific
honorific_dict = {}

for honorific in set(honorifics):
    honorific_dict[honorific + '_num'] = 0
    honorific_dict[honorific + '_sum'] = 0

for index, row in train_data.iterrows():
    if pd.notnull(row['Age']):
        honorific_dict[row['Honorific'] + '_num'] += 1
        honorific_dict[row['Honorific'] + '_sum'] += row['Age']

for honorific in set(honorifics):
    honorific_dict[honorific + '_avg'] = honorific_dict[honorific + '_sum']/honorific_dict[honorific + '_num']

for index, row in train_data.iterrows():
    if not pd.notnull(row['Age']):
        train_data['Age'][index] = honorific_dict[row['Honorific'] + '_avg']


Y = train_data['Survived']
X = train_data.drop('Survived', axis=1)

# tree = DecisionTreeClassifier(max_depth=5)
# tree.fit(X, Y)
