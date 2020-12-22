import csv
import time

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def extract_ticket_number(tickets):
    tktNum = []
    for ticket in tickets:
        if ticket == 'LINE':
            tktNum.append(-1)
        else:
            splits = ticket.split()
            tktNum.append(int(splits[len(splits) - 1]))

    return tktNum


def encode_embarkation(dataset):
    embarked_1hot_encoder = OneHotEncoder()
    transformed_embarked = embarked_1hot_encoder.fit_transform(dataset[['Embarked']])

    for i in range(len(embarked_1hot_encoder.categories_[0])):
        dataset[embarked_1hot_encoder.categories_[0][i]] = transformed_embarked.toarray()[:, i]


def extract_honorific(dataset):
    names = dataset['Name']
    honorifics = []
    for name in names:
        start = name.find(',') + 1
        end = name.find('.')
        honorifics.append(name[start:end])

    return honorifics


# Read train data
train_data = pd.read_csv("./data/train.csv", index_col='PassengerId')

# Read test data
test_data = pd.read_csv("./data/test.csv")
test_index_column = test_data['PassengerId']
test_data = test_data.drop('PassengerId', axis=1)
# Fix null values

# null embarked --> mode
embarked = train_data['Embarked']
embarked.fillna(embarked.mode()[0], inplace=True)

# drop cabin
train_data = train_data.drop('Cabin', axis=1)
test_data = test_data.drop('Cabin', axis=1)


# extracting honorifics
honorifics = extract_honorific(test_data)
test_data['Honorific'] = honorifics

honorifics = extract_honorific(train_data)

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
    honorific_dict[honorific + '_avg'] = honorific_dict[honorific + '_sum'] / honorific_dict[honorific + '_num']

# null age --> median
for index, row in train_data.iterrows():
    if not pd.notnull(row['Age']):
        train_data['Age'][index] = honorific_dict[row['Honorific'] + '_avg']

for index, row in test_data.iterrows():
    if not pd.notnull(row['Age']):
        test_data['Age'][index] = honorific_dict[row['Honorific'] + '_avg']

train_data = train_data.drop('Honorific', axis=1).drop('Name', axis=1)
test_data = test_data.drop('Honorific', axis=1).drop('Name', axis=1)

sex_to_num = OrdinalEncoder()
train_data[['Sex']] = sex_to_num.fit_transform(train_data[['Sex']])
test_data[['Sex']] = sex_to_num.fit_transform(test_data[['Sex']])

# for now I assign -1 to LINES (time problem)
# for now I don't care about tkt pref (time problem)
tickets = train_data['Ticket']
tktNum = extract_ticket_number(tickets)
train_data['TktNum'] = tktNum

tickets = test_data['Ticket']
tktNum = extract_ticket_number(tickets)
test_data['TktNum'] = tktNum

train_data = train_data.drop('Ticket', axis=1)
test_data = test_data.drop('Ticket', axis=1)

encode_embarkation(train_data)
encode_embarkation(test_data)

train_data = train_data.drop('Embarked', axis=1)
test_data = test_data.drop('Embarked', axis=1)

Y = train_data['Survived']
X = train_data.drop('Survived', axis=1)

print(train_data.info())

fare_median = train_data['Fare'].median()

test_data['Fare'].fillna(value=fare_median, inplace=True)

print(test_data.info())

forest_clf = RandomForestClassifier(random_state=42, max_depth=5, criterion="entropy")
# starting time
start = time.time()
forest_clf.fit(X, Y)
# end time
end = time.time()
# total time taken
print(f"random forest learning time was {end - start}")

survived_prediction = forest_clf.predict(test_data)

with open("./data/prediction.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["PassengerId", "Survived"])
    for p_id, prediction in zip(test_index_column, survived_prediction):
        writer.writerow([p_id, prediction])
