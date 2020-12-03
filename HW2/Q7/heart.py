import pandas as pd
from sklearn.model_selection import train_test_split

# Read train data
data = pd.read_csv("./data/heart.csv")
# print(data.info())

train_data, test_data = train_test_split(data, test_size=0.2, random_state=43)

