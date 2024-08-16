import numpy as np

X = np.random.normal(loc=0.0, scale=1.0, size=(100,50))
M, N = X.shape
K = 10
Omega = np.random.normal(loc=0.0, scale=1.0, size=(N,K))
#print(X)
#Y = X * Omega
Y = np.dot(X, Omega)
print(Y.shape)
Q,R = np.linalg.qr(Y,mode='reduced')
print(Q.shape)

print(K)
print(M)
print(N)
print("ss")

B = np.transpose(Q).dot(X)


U,S,V = np.linalg.svd(B)
print(U.shape)
uapp = Q.dot(U)
print(uapp.shape)