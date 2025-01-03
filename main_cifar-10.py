import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets
import sklearn.preprocessing
from pandas import read_csv
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import csv
import time
import pickle
matplotlib.use("TkAgg")

def plotimage(batch1, image_index):
    print(type(batch1))
    print(batch1[b'data'][0])
    print(batch1[b'data'][0].shape)

    # Separate RGB lists (values between 0 and 255)
    r_values = batch1[b'data'][image_index][0:1024]
    print(r_values.shape)
    g_values = batch1[b'data'][image_index][1024:1024*2]
    print(g_values.shape)
    b_values = batch1[b'data'][image_index][1024*2:]
    print(b_values.shape)

    # Normalize the RGB values to the range [0, 1]
    r_normalized = [r / 255 for r in r_values]
    g_normalized = [g / 255 for g in g_values]
    b_normalized = [b / 255 for b in b_values]

    # Combine the normalized RGB values into a 1xN array where N is the number of colors
    colors = np.array([list(zip(r_normalized, g_normalized, b_normalized))]).reshape(32,32,3)
    print(colors.shape)
    # Use imshow to plot the array of colors
    plt.imshow(colors, aspect='auto')

    # Hide the axes and display the plot
    plt.gca().set_axis_off()
    plt.show()

    # Show singular image
    #plt.imshow(image.reshape(32, 32, 3), cmap="Greys")
    #plt.show()
    #plt.close()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch1 = unpickle("cifar-10-batches-py/data_batch_1")
test_batch = unpickle("cifar-10-batches-py/test_batch")

image = batch1[b'data'][0][0:]
x_train = batch1[b'data'][:][0:]
y_train = np.array(test_batch[b'labels'][:][0:])
print(x_train.shape)
print(image.shape)


x_train_rowvector = np.reshape(x_train, (-1, 32*32))
x_train_colvector = np.copy(x_train_rowvector).T
#x_test_rowvector = np.reshape(x_test, (-1, 28*28))
#x_test_colvector = np.copy(x_test_rowvector).T
# Take small sample of 2000 training images
x_train_colvector_sample2000 = x_train_colvector[:, :2000]
y_train_sample2000 = y_train[:2000]

# Plot image
plotimage(batch1, 0)

# Original LA
start = time.time()
u_org, s_org, v_org = np.linalg.svd(x_train_colvector_sample2000, full_matrices=False)
done = time.time()
elapsed = done - start
print(f"Time elapsed for original matrix: {elapsed}")


# RNLA
start = time.time()
X = x_train_colvector_sample2000
print("X.shape")
print(X.shape)
M, N = X.shape
K = 100
Omega = np.random.normal(loc=0.0, scale=1.0, size=(N,K))
print("Omega shape")
print(Omega.shape)
Y = X@Omega
print("Y shape")
print(Y.shape)
Q,R = np.linalg.qr(Y, mode='reduced')
print("Q shape")
print(Q.shape)
Q_transpose = np.transpose(Q)
B = Q_transpose@X
print("B.shape")
print(B.shape)
#x_train_colvector_sample2000 = B

# Calculate u, s, v
u, s, v = np.linalg.svd(B, full_matrices=False)
done = time.time()
elapsed = done - start
print(f"Time elapsed for random projected matrix: {elapsed}")

<<<<<<< HEAD
=======
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch1 = unpickle("cifar-10-batches-py/data_batch_1")
print(type(batch1))
print(batch1[b'data'][0])
print(batch1[b'data'][0].shape)
image = batch1[b'data'][0][0:]
print(image.shape)


image_index = 2
# Separate RGB lists (values between 0 and 255)
r_values = batch1[b'data'][image_index][0:1024]
print(r_values.shape)
g_values = batch1[b'data'][image_index][1024:1024*2]
print(g_values.shape)
b_values = batch1[b'data'][image_index][1024*2:]
print(b_values.shape)

# Normalize the RGB values to the range [0, 1]
r_normalized = [r / 255 for r in r_values]
g_normalized = [g / 255 for g in g_values]
b_normalized = [b / 255 for b in b_values]

# Combine the normalized RGB values into a 1xN array where N is the number of colors
colors = np.array([list(zip(r_normalized, g_normalized, b_normalized))]).reshape(32,32,3)
print(colors.shape)
# Use imshow to plot the array of colors
plt.imshow(colors, aspect='auto')

# Hide the axes and display the plot
plt.gca().set_axis_off()
plt.show()
exit(0)
>>>>>>> c5f7184 (s)

# Plot image
svd_image = Q@u
image = svd_image[:,0]
print(image.reshape(32, 32))
# Show singular image
plt.imshow(image.reshape(32, 32), cmap="Greys")
plt.show()
plt.close()

# Set all singular values greater than the first two to 0
#print(s.shape[0])
for i in range(2, s.shape[0]):
    s[i] = 0
# Calculate the reduced dimensions with svd
svd_cords = np.diag(s) @ v
# U[:x, :x] @ np.diag(S[:x, :x]) @ V[:x,:x]
svd_image = u @ svd_cords


print(svd_cords.shape)
print(y_train_sample2000.shape)
svd_list= [0] * 10
for i in range(10):
    svd_list[i] = svd_cords.T[y_train_sample2000 == i]
print(len(y_train_sample2000 == i))
print(len(svd_list))
with open('your_file.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(svd_list)

COLORS = ["red", "blue", "green", "yellow", "darkviolet",
          "maroon", "greenyellow", "hotpink", "black", "cyan"]
fig, ax = plt.subplots()
for i in range(10):
    # Get the pca array corresponding to the current label
    svd_current_label = svd_list[i]
    ax.scatter(svd_current_label[:, 0], svd_current_label[:, 1],
               c=COLORS[i], label=str(i))

ax.legend()
plt.show()
#https://github.com/DanielY1783/mnist_svd/blob/master/mnist.ipynb
svd_mean_list = [0] * 10
for i in range(10):
    svd_mean_list[i] = np.mean(svd_list[i], axis=0)

COLORS = ["red", "blue", "green", "yellow", "darkviolet", 
          "maroon", "greenyellow", "hotpink", "black", "cyan"]
fig, ax = plt.subplots()
for i in range(10):
    # Get the pca array corresponding to the current label
    svd_current_label = svd_mean_list[i]
    ax.scatter(svd_current_label[0], svd_current_label[1],
               c=COLORS[i], label=str(i))

ax.legend()
plt.show()
