import pandas as pd

df = pd.read_csv("data/covid.csv")
# print(df)

# print(df.columns)
# print(len(df))
# print(df.isna().sum())

by = df["birth_year"]

print(by.max())

# print(by.mean())
# print(by.fillna(0).to_numpy().mean())

# three ways 1. by.mean() and desc 2. numpy.mean() what it does with nan & fillna 3. scikit learn transform use median instead of zero in fillna check hands on ml