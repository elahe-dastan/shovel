import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ..Q1.k_means import kmeans

img = mpimg.imread('sample_img1.png')
# converting image to two dimensional
two_d_image = img.reshape(-1, 3)

obj = kmeans(two_d_image, 64, 40)
clusters, centers = obj.cluster()

for i in range(two_d_image.shape[0]):
    two_d_image[i] = centers[clusters[i]]

compressed_image = img.reshape(img.shape[0], img.shape[1], 3)
plt.imshow(compressed_image)
plt.show()
