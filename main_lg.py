import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel

# Generate some synthetic data
np.random.seed(42)
n = 30
X = np.linspace(0, 5, n)[:, None]
y_true = np.sin(X).ravel()
y = y_true + 0.3 * np.random.randn(n)

# -----------------------------
# Ordinary Ridge Regression (linear features)
# -----------------------------
ridge = Ridge(alpha=1.0, fit_intercept=True)
ridge.fit(X, y)
y_pred_lin = ridge.predict(X)

# -----------------------------
# Kernel Ridge Regression (Kimeldorf–Wahba connection)
# -----------------------------
gamma = 1.0
K = rbf_kernel(X, X, gamma=gamma)   # kernel matrix
alpha = np.linalg.solve(K + 1.0 * np.eye(n), y)  # regularized solution
y_pred_kernel = K @ alpha

# -----------------------------
# Plot results
# -----------------------------
plt.scatter(X, y, label="Noisy data", color="black")
plt.plot(X, y_true, label="True function", color="green")
plt.plot(X, y_pred_lin, label="Linear Ridge", color="blue")
plt.plot(X, y_pred_kernel, label="Kernel Ridge (Kimeldorf-Wahba)", color="red")
plt.legend()
plt.title("Kimeldorf-Wahba theorem demo: linear ridge vs kernel ridge")
plt.show()