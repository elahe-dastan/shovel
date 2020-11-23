import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

# Read data
train_data = pd.read_csv("data/train.csv", index_col='PassengerId')
print(train_data.info())

# Fix null values

# null embarked --> mode
embarked = train_data['Embarked']
embarked.fillna(embarked.mode()[0], inplace=True)

# drop cabin
train_data = train_data.drop('Cabin', axis=1)

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

# null age --> median
for index, row in train_data.iterrows():
    if not pd.notnull(row['Age']):
        train_data['Age'][index] = honorific_dict[row['Honorific'] + '_avg']

train_data = train_data.drop('Honorific', axis=1).drop('Name', axis=1)

sex_to_num = OrdinalEncoder()
train_data[['Sex']] = sex_to_num.fit_transform(train_data[['Sex']])

# for now I assign -1 to LINES (time problem)
# for now I don't care about tkt pref (time problem)
tickets = train_data['Ticket']
tktNum = []
for ticket in tickets:
    if ticket == 'LINE':
        tktNum.append(-1)
    else:
        splits = ticket.split()
        tktNum.append(int(splits[len(splits) - 1]))

train_data['TktNum'] = tktNum

train_data = train_data.drop('Ticket', axis=1)

print(train_data.info())


Y = train_data['Survived']
X = train_data.drop('Survived', axis=1)

# tree = DecisionTreeClassifier(max_depth=5)
# tree.fit(X, Y)
