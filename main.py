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
print(X.shape)
M, N = X.shape
K = 3
Omega = np.random.normal(loc=0.0, scale=1.0, size=(N,K))
print("Omega shape")
print(Omega.shape)
Y = X@Omega
print("Y shape")
print(Y.shape)
Q,R = np.linalg.qr(Y, mode='reduced')
print("Q shape")
print(Q.shape)

print(K)
print(M)
print(N)
print("ss")
Q_transpose = np.transpose(Q)
B = Q_transpose@X
print("B shape")
print(B.shape)
U_hat,sum_hat,v_hat = np.linalg.svd(B, full_matrices=False)
print("U hat shape")
print(U_hat.shape)
U = Q@U_hat
print("U shape")
print(U.shape)

print('matrix U has {} rows, {} columns\n'.format(*U.shape))
print('here are the first 5 rows.')

print('{}'.format(pd.DataFrame(U).head(5)))



'''
A = np.array([
    [1,2,3],
    [4,5,6]
])
print(A.shape)
'''