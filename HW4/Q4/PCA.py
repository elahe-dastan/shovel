from sklearn.decomposition import PCA

f = open("glove.6B/glove.6B.50d.txt", "r")
lines = {}
for i in range(1000):
    line = f.readline()
    
