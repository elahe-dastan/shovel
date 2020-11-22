import pandas as pd
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

Y = train_data['Survived']
X = train_data.drop('Survived', axis=1)

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X, Y)

