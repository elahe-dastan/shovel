import matplotlib.pyplot as plt
import pandas as pd

# illustrating dataset 1
data1 = pd.read_csv("Dataset1.csv")
X1 = data1["X"]
Y1 = data1["Y"]

plt.scatter(X1, Y1)
plt.title("Dataset1 Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# illustrating dataset 2
data2 = pd.read_csv("Dataset2.csv")
X2 = data2["X"]
Y2 = data2["Y"]

plt.scatter(X2, Y2)
plt.title("Dataset2 Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
