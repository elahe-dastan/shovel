import matplotlib.pyplot as plt
import pandas as pd

# illustrating dataset1
data1 = pd.read_csv("Dataset1.csv")
X = data1["X"]
Y = data1["Y"]

plt.scatter(X, Y)
plt.title("Dataset1 Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
