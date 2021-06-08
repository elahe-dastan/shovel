import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import k_means

img = mpimg.imread('Q2-testcase.png')
# converting image to two dimensional
two_d_image = img.reshape(-1, 3)

k = 4
start_time = time.time()
obj = k_means.kmeans(two_d_image, k, 40)
clusters, centers = obj.cluster()

for i in range(two_d_image.shape[0]):
    two_d_image[i] = centers[clusters[i]]
end_time = time.time()
print("it took me {} seconds to do the compression for {} clusters".format(end_time - start_time, k))
compressed_image = img.reshape(img.shape[0], img.shape[1], 3)
plt.imshow(compressed_image)
plt.show()
