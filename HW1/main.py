import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import numpy as np

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
# print(preprocessed_df)

# plt.hist(df['birth_year'])
# plt.show()
#
# plt.hist(df['infected_by'])
# plt.show()

# plt.scatter(df["birth_year"], df["confirmed_date"])
# plt.show()

pd.plotting.scatter_matrix(df, alpha=0.2)


# detect and remove outliers for 'confirmed_date'
# Note: the exact same thing below can be done for 'birth_year' and 'infected_by' columns too
cd_mean = df['confirmed_date'].mean()
cd_std = df['confirmed_date'].std()

cd_z_score = (df['confirmed_date'] - cd_mean) / cd_std

cd_median = df['confirmed_date'].median()

cd = np.where(cd_z_score.abs() > 3, cd_median, df['confirmed_date'])

