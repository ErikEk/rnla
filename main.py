import numpy as np
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.stats
import sklearn.datasets
import sklearn.preprocessing


iris = sklearn.datasets.load_iris()

X = iris.data #np.random.normal(loc=0.0, scale=1.0, size=(100,50))
M, N = X.shape
K = 10
Omega = np.random.normal(loc=0.0, scale=1.0, size=(N,K))

Y = np.dot(X, Omega)
print(Y.shape)
Q,R = np.linalg.qr(Y,mode='reduced')
print(Q.shape)

print(K)
print(M)
print(N)
print("ss")

B = np.transpose(Q).dot(X)

U_hat,sum_hat,v_hat = np.linalg.svd(B)
print(U_hat.shape)
U = Q.dot(U_hat)

print('matrix U has {} rows, {} columns\n'.format(*U.shape))
print('here are the first 5 rows.')

print('{}'.format(pd.DataFrame(U).head(5)))