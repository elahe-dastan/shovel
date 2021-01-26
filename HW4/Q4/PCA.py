from sklearn.decomposition import PCA
import numpy as np

f = open("glove.6B/glove.6B.50d.txt", "r")
lines = {}
wordss = []
a = np.zeros((1000, 50))
for i in range(1000):
    line = f.readline()
    if line.endswith('\n'):
        line = line[:-1]
    words = line.split(" ")
    a[i] = words[1:]
    wordss.append(words[0])
    lines[words[0]] = words[1:]
# print(lines)
pca = PCA(n_components=2)
b = pca.fit_transform(a)
for i in range(1000):
    print(wordss[i])
    print(b[i])
# print(b)
