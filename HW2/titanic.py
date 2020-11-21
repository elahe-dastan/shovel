import pandas as pd

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
