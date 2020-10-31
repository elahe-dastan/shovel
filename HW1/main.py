import pandas as pd

df = pd.read_csv("data/covid.csv")
# print(df)

# print(df.columns)
# print(len(df))
print(df.isna().sum())

by = df["birth_year"]

# print(by.mean)
# print(by.fillna(0).to_numpy().mean())
