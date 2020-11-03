import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

df = pd.read_csv("data/covid.csv")
# print(df)

# print(df.columns)
# print(len(df))
# print(df.isna().sum())

by = df["birth_year"]

# print(by.max())

# mean
# first
# print(by.mean())

# second
# print(by.fillna(0).to_numpy().mean())

# std
# print(by.std())

df['confirmed_date'] = pd.to_datetime(df['confirmed_date']).astype(int) / 10**9

numerical_columns = ["birth_year", "infected_by", "confirmed_date"]
nominal_columns = ["id", "sex", "country", "region", "infection_reason", "state"]

nan_transformer = ColumnTransformer([
    ("numerical", SimpleImputer(strategy="median"), numerical_columns),
    ("nominal", SimpleImputer(strategy="most_frequent"), nominal_columns),
])

preprocessed_df = nan_transformer.fit_transform(df)
print(preprocessed_df)
