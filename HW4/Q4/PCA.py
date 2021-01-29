from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

f = open("glove.6B/glove.6B.300d.txt", "r")
subset_size = 20
dimension = 300
word_vector_dict = {}
words = [None]*subset_size
vectors = np.zeros((subset_size, dimension))
for i in range(1000):
    f.readline()
for i in range(subset_size):
    line = f.readline()
    if line.endswith('\n'):
        line = line[:-1]
    word_vectors = line.split(" ")
    vectors[i] = word_vectors[1:]
    words[i] = word_vectors[0]
    word_vector_dict[word_vectors[0]] = word_vectors[1:]
# print(lines)
pca = PCA(n_components=2)
two_dimension = pca.fit_transform(vectors)

fig, ax = plt.subplots()
for i in range(subset_size):
    ax.scatter(vectors[i][0], vectors[i][1])
    ax.annotate(words[i], (vectors[i][0], vectors[i][1]))
    print(words[i])
    print(two_dimension[i])

plt.show()
# print(b)
