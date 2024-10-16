import numpy as np
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.stats
import sklearn.datasets
import sklearn.preprocessing
from pandas import read_csv

#iris = sklearn.datasets.load_iris()
#print(iris)

#X = iris.data #np.random.normal(loc=0.0, scale=1.0, size=(100,50))
X = read_csv('Food_contents_2024.csv')
print(X.head(5))
exit(0)
X = X.apply(pd.to_numeric, errors='coerce')
print(X.head(5))
X = X.fillna(0.0)
X.to_csv("Food_formatted.csv")

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


U_iris = U

# Projecting the data onto the right singular vectors
#P_iris = df_iris.to_numpy().dot(Vt_iris.T)

idx_setosa = np.where(X.columns=='Energy (kcal)')[0]
setosa_x = U_iris[idx_setosa, 0]
setosa_y = U_iris[idx_setosa, 1]
idx_versicolor = np.where(X.columns=='Protein (g)')[0]
idx_virginica = np.where(X.columns=='Carbohydrate (g)')[0]
print(setosa_x.shape)
versicolor_x = U_iris[idx_versicolor, 0]
versicolor_y = U_iris[idx_versicolor, 1]

virginica_x = U_iris[idx_virginica, 0]
virginica_y = U_iris[idx_virginica, 1]

fig = plt.figure(figsize=(7.0,5.5))
ax = fig.add_subplot(111)
plt.scatter(setosa_x,
            setosa_y,
            marker='o',
            color='#66c2a5',
            label='Iris-setosa',
            zorder=1000)
plt.scatter(versicolor_x,
            versicolor_y,
            marker='D',
            color='#fc8d62',
            label='Iris-versicolor',
            zorder=1000)

plt.scatter(virginica_x,
            virginica_y,
            marker='^',
            color='#8da0cb',
            label='Iris-virginica',
            zorder=1000)

ax.set_xlabel(r'singular value $\sigma_{1}$')
ax.set_ylabel(r'singular value $\sigma_{2}$')
plt.grid(alpha=0.6, zorder=1)
ax.set_facecolor('0.98')
plt.tight_layout()
plt.show()