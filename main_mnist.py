import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets
import sklearn.preprocessing
from pandas import read_csv
from keras.datasets import mnist
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import csv
import time
matplotlib.use("TkAgg")

# Load in mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Reshape to each image to a row vector and column vector
x_train_rowvector = np.reshape(x_train, (-1, 28*28))
x_train_colvector = np.copy(x_train_rowvector).T
x_test_rowvector = np.reshape(x_test, (-1, 28*28))
x_test_colvector = np.copy(x_test_rowvector).T
# Take small sample of 2000 training images
x_train_colvector_sample2000 = x_train_colvector[:, :2000]
y_train_sample2000 = y_train[:2000]
# Take small sample of 200 testing images
x_test_colvector_sample200 = x_test_colvector[:, :200]
y_test_sample200 = y_test[:200]

print(x_train.shape)
print(x_test.shape)
print(x_train_rowvector.shape)
print(x_test_rowvector.shape)

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

# Calculate u, s, v
u, s, v = np.linalg.svd(B, full_matrices=False)
done = time.time()
elapsed = done - start
print(f"Time elapsed for random projected matrix: {elapsed}")

start = time.time()
u_org, s_org, v_org = np.linalg.svd(x_train_colvector_sample2000, full_matrices=False)
done = time.time()
elapsed = done - start
print(f"Time elapsed for original matrix: {elapsed}")


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



image = x_train_colvector_sample2000[:,0]
# Show singular image
plt.imshow(image.reshape(28, 28), cmap="Greys")
plt.show()


#image = svd_image[:,0]
#kk = Q@u
test_svd = u_org @ np.diag(s_org) @ v_org
image =  test_svd[:,0]
# Show singular image
plt.imshow(image.reshape(28, 28), cmap="Greys")
plt.show()
