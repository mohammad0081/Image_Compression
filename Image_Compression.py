import numpy as np
import matplotlib.pyplot as plt

# Load an image of a bird
original_img = plt.imread('sky.jpg')
plt.imshow(original_img)
plt.show()


def init_centroids(X, k):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:k]]
    return centroids


def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K) :
        points = X[idx == k]

        if len(points) > 0:
            centroids[k] = np.mean(points, axis= 0)

    return centroids


def find_closest_centroid(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distances = []
        for j in range(centroids.shape[0]):
            norm_i_j = np.linalg.norm(X[i] - centroids[j])
            distances.append(norm_i_j)

        idx[i] = np.argmin(distances)

    return idx


def run_k_means(X, initial_centroids, max_iters=100):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)

    for i in range(max_iters):
        print(f'Epoch : {i} | K = {K}')
        idx = find_closest_centroid(X, centroids)
        centroids = compute_centroids(X, idx, K)

    return centroids, idx


# Divide by 255 so that all values are in the range 0 - 1
original_img = original_img / 255

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 1

# Using the function you have implemented above.
initial_centroids = init_centroids(X_img, K)

# Run K-Means - this takes a couple of minutes
centroids, idx = run_k_means(X_img, initial_centroids, max_iters)

# Represent image in terms of indices
X_recovered = centroids[idx, :]

# Reshape recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape)

plt.imshow(X_recovered)
plt.show()

# Concatenate the images along the second axis (i.e., width)
concatenated_image = np.concatenate((original_img, X_recovered), axis=1)

# Create a figure and a set of subplots
fig, ax = plt.subplots()


# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Display an image on the axes
ax.imshow(concatenated_image)

# Draw a red line between the images
plt.plot([1280, 1280], [0, 720], color='red')

plt.axis('off')

# Save the figure
plt.savefig('final_image.png', bbox_inches='tight', pad_inches=0)

# Display the figure
plt.show()